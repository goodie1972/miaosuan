# 妙算（MiaoSuan）调优引擎升级路线 —— OOS 验证 / 采样器可切换 / LLM 语义探索

| 项目 | 内容 |
|---|---|
| 文档类型 | 未来升级设计（可执行路线图） |
| 作者 | 架构师 |
| 版本 | v1.0 |
| 日期 | 2026-09-16 |
| 上游输入 | 2026-09-16 用户咨询「Optuna TPE + 全样本回测的效率/可靠度、市场替代方案、AI 判定选项」 |
| 关联代码 | `miaosuan/src/miaosuan/tune/engine.py`（`299a9e3` 起，自动挖参 + `evaluate_trial` 接线已落地） |
| 性质 | **设计 + 路线图，不含业务代码；不修改现有代码** |

> 本文把「调优引擎下一步」沉淀为可执行路线，分三档（P0/P1/P2）。
> 核心判断：**可靠度的最大瓶颈不是采样器，而是"只在样本内挑最高 composite"**。
> 因此 P0 永远是 OOS walk-forward，采样器可切换与 LLM 增强都是"锦上添花"，且必须压在 P0 之上做。

---

## 0. 结论摘要（先说答案）

| # | 问题 | 路线结论 |
|---|---|---|
| 1 | TPE 在当前 3 参数混合空间上够不够？ | **够，且接近默认最优**。`engine.py` 现用 TPE，3 旋钮（float/int/choice）正是 TPE 的强项。先跑通 + 加 OOS，再谈换采样器。 |
| 2 | 全样本回测慢，瓶颈在哪？ | **在评估成本，不在采样器**。换采样器救不了"每个 trial 分钟级"。优化顺序：OOS 验证 → 预算控制 → （可选）粗筛精调。 |
| 3 | 可靠度最大风险？ | **样本内过拟合**。现 `composite = 0.6×sharpe − 0.4×min(mdd/100,1)` 是防回撤型护栏，但不防"纯 Sharpe 尖峰型"过拟合。必须 OOS walk-forward。 |
| 4 | 市场上有没有更好选择？ | 有，但"更好"分场景：预算 <200 试 **GPSampler**；旋钮间强交互试 **CmaEsSampler**；纯基准对照用 **QMCSampler/RandomSampler**。3 参数空间下 TPE 已近乎最优。 |
| 5 | 加入 AI 判定有何选项？ | 2025 前沿：AgentHPO（PMLR）、NNGPT（ICCVW）、XAutoLM（EMNLP）。**成熟度低、需 LLM 后端**，不替代 TPE 主干；定位为"**语义探索增强器**"（LLM 出下一组参数假设，TPE 吃 suggestion）。 |
| 6 | 执行顺序？ | **P0 OOS walk-forward → P1 采样器可切换 + 预算/粗筛 → P2 LLM 探索增强**。P2 之前必须 P0 已固化，否则 LLM 给的"讲道理"参数更易过拟合。 |

---

## 1. 现状基线（`engine.py` 已落地）

### 1.1 已实现的"语义旋钮"
`SEMANTIC_SEARCH_SPACES` 三旋钮，键名与 `Semantics` 字段逐字一致：

| 旋钮 | 类型 | 缺省 | 作用 | 落到回测的实参 |
|---|---|---|---|---|
| `neutral_band` | float | 0.05 | 中性带：\|pos\|<值 置 0 不交易 | `run_full_backtest(neutral_band=…)` |
| `roll_window` | int | 500 | 因子归一化窗口（bar 数） | `factor_roll_window` → `StackVM(roll_window)` |
| `long_only` | choice | 0 | 是否只做多 | `long_only`（choice 串 → `_as_bool`） |

`load_param_space_from_spec` 读 spec 后调 `mine_param_spaces`，前端留空时自动挖这三项。

### 1.2 目标函数
`score_trial` 支持 `sharpe / sortino / calmar / composite`，webui 默认走 `composite`：

```
composite = 0.6 × sharpe − 0.4 × min(mdd/100, 1.0)
```

这是**回撤型护栏**（惩罚 mdd 极端的曲线），但**不防"高 sharpe 尖峰"型过拟合** —— 即曲线在样本内偶然冒尖、OOS 掉坑。这正是 P0 要解决的。

### 1.3 已知边界
- 评估 = 全样本回测，分钟级；3 参数 × TPE 默认 budget 下 trial 数不小，**预算是墙钟第一瓶颈**。
- 现无 OOS 段、无采样器对照、无 LLM 探索；`objective` 直接 `study.optimize`，单一 TPE。

---

## 2. P0 —— 样本外 / walk-forward 验证（最高优先）

> **先做这个。** 它把"挑样本内最高"升级为"样本内寻优 + 保留段验证"，是可靠度提升最大的一步，比换任何采样器都重要。

### 2.1 数据切分（复用 `data/split.py` 的三段切分 + holdout 封印）
- **训练段（调优段）**：TPE 在此段寻优，`composite` 只在训练段计算。
- **保留段（OOS / holdout）**：一次性封印（`data/split.py` 已有 `holdout` 语义，对齐 §6.1 `HoldoutSealedError`）。只在训练段完成后**单次**评估，**不得**用 OOS 反复调参。
- **walk-forward（可选增强）**：把保留段进一步切成多个滚动窗（如 5 折），每折"前段调优 + 后段验证"，看 composite 在折间是否**稳定**而非尖峰。

### 2.2 执行步骤
1. 在 `TuneConfig` 增加 `oos_ratio`（默认 0.3）与 `walk_forward_folds`（默认 1，1 即单次 OOS；>1 即滚动验证）。
2. `evaluate_trial` 拆成两段：训练段回测算 `composite`（TPE 目标）；OOS 段回测算 `composite_oos`（记录但不进目标，避免 peek）。
3. `TuneResult` 扩展字段：`best_params_insample`、`composite_oos`、`oos_composite_ratio`（= OOS composite / 样本内 composite）、`walk_forward_var`（折间方差）。
4. **过拟合判定线**：`oos_composite_ratio < 0.6` 或 `walk_forward_var > 阈值` → 标 `RESEARCH_ONLY`（对齐 §6.4 软失败），提示"样本外衰减显著"。
5. webui 调优结果页显示「样本内 best / OOS composite / 折间方差」三行，红字提示衰减。

**验收**：能产出一份「样本内 best 参数在 OOS 上衰减 40%+」的案例，证明 P0 拦截了样本内假最优。

---

## 3. P1 —— 采样器可切换 + 预算控制（锦上添花，压在 P0 之上）

> 前提：P0 已固化。所有采样器**统一用同一套 objective（composite）+ OOS 验证**，才公平比较。

### 3.1 采样器矩阵（2025 格局）

| 采样器 | 适用 | 相对 TPE 的取舍 | MiaoSuan 建议 |
|---|---|---|---|
| **TPE**（现默认） | 混合类型、100~1000 budget | 稳、可解释 | 基准，保留 |
| **GPSampler**（GP） | 低预算 ≤200、纯数值 | 样本效率最高（O(n³)） | budget<200 时对照 |
| **CmaEsSampler**（进化） | 连续强交互 | 联合最优更强；**不支持 choice** | 仅当 roll_window↔neutral_band 强相关时对照（需把 long_only 固定） |
| **QMCSampler**（准随机） | 快速撒网 baseline | 无"学习"，纯基准 | 对照组（验证 TPE 是否真占便宜） |
| **SMAC3**（4.2+） | 大规模并行 | overkill | 不做 |
| **c-TPE**（IJCAI'25） | 有约束连续 | 约束弱用不上 | 不做 |

### 3.2 执行步骤
1. `TuneConfig.sampler: str = "tpe"`，枚举 `tpe / gps / cmaes / qmc / random`；`TuneEngine` 内按枚举构造对应 `study.sampler`，**objective 不变**（仍走 `score_trial` composite + P0 的 OOS）。
2. 对照实验：同一 budget 下跑 `tpe` vs `qmc` vs `gps`，输出「trial 数 → best composite」收敛曲线（画进 webui 报告），直观看 TPE 是否压住随机。
3. **预算控制**：复用 §6.2 三档（quick/standard/deep）墙钟硬约束，落到 OOS 验证上——quick 档可只跑 1 折 OOS，deep 档跑 5 折 walk-forward。
4. （可选）粗筛精调两阶段：先用 30 trial 的 QMC 撒网定位热点区域，再以热点中心为起点跑 TPE，降低总墙钟。

**验收**：webui 能切采样器并出收敛曲线；同一 budget 下至少有一组对照证明「换采样器是否值得」。

---

## 4. P2 —— LLM 语义探索增强（PoC，不替代主干）

> 定位：**LLM 出"下一组参数假设"，Optuna 吃 suggestion，TPE 仍是数值收敛主干。** 不引入生产级 LLM 依赖；需 LLM 后端，成本/延迟由调用方承担。

### 4.1 2025 参照（做研究对照，不直接搬）

| 方法 | 机制 | 证据 | 对 MiaoSuan 的可借鉴点 |
|---|---|---|---|
| **AgentHPO**（PMLR'25） | LLM agent（Creator+Executor）+ 经验记忆 | GPT-4 第 10 trial 比人类最佳高 1.52% | "读 spec + 指标 → 生成下一组参数"的探索范式 |
| **NNGPT**（ICCVW'25） | 微调 LLM 做"一次 HPO" | HPO RMSE 0.60 vs Optuna 0.64 | 需 HPO 训练语料，冷启动不可行 → 仅记录 |
| **XAutoLM**（EMNLP'25） | 元学习 + 跨任务经验库 | 跨 benchmark 稳 | 依赖经验库冷启动，不友好 → 仅记录 |

### 4.2 执行步骤（PoC）
1. 新增 `tune/llm_suggest.py`：`suggest_next(spec, trial_history, metrics_summary) -> dict[str, value]`，用 LLM 读 spec 语义 + 历史 trial 的 composite/sharpe/mdd，**有解释性地**输出下一组三旋钮取值。
2. 接进 Optuna：`study.enqueue_trial(suggestions)` 或每 trial 后把 LLM 建议作为一条 `suggestion` 注入；TPE 继续主导数值收敛，LLM 只在"语义方向"上探。
3. **护栏不放松**：objective 仍 `composite` + P0 的 OOS 验证。LLM 建议若 OOS 衰减超阈值，直接打回 RESEARCH_ONLY。
4. 指标：LLM 增强组 vs 纯 TPE 组，比"达到相同 OOS composite 所需 trial 数"。

**验收**：LLM 建议未让 OOS 衰减变差，且 trial 数下降 ≥ 20%。

---

## 5. 落地文件清单（不写业务代码，仅映射到现有/待加文件）

| 改动 | 文件 | 档 |
|---|---|---|
| `TuneConfig` 加 `oos_ratio` / `walk_forward_folds` | `src/miaosuan/tune/engine.py`（dataclass） | P0 |
| `evaluate_trial` 拆训练段/OOS 段 | `src/miaosuan/tune/engine.py` | P0 |
| `TuneResult` 加 `composite_oos` / `oos_composite_ratio` / `walk_forward_var` | `src/miaosuan/tune/engine.py` | P0 |
| OOS 衰减判定 + RESEARCH_ONLY 标记 | 复用 §6.4 `gate/verdict.py` 语义 | P0 |
| `TuneConfig.sampler` 枚举 + `TuneEngine` 构造对应 study | `src/miaosuan/tune/engine.py` | P1 |
| 采样器收敛曲线 | `src/miaosuan/report/`（html） | P1 |
| 三档预算落到 OOS 折数 | 复用 §6.2 `search/budget.py` | P1 |
| LLM 建议注入 | 新增 `src/miaosuan/tune/llm_suggest.py` | P2 |

---

## 6. 总执行顺序与里程碑

```
P0  OOS walk-forward（先做，可靠性地基）
      ├─ M1: TuneConfig.oos_ratio + evaluate_trial 两段回测
      ├─ M2: TuneResult OOS 字段 + 衰减判定 → RESEARCH_ONLY
      └─ M3: webui 显示 样本内/OOS/折间方差
  ↓
P1  采样器可切换 + 预算（压在 P0 上）
      ├─ M4: TuneConfig.sampler 枚举 + study 构造
      ├─ M5: 收敛曲线对照（tpe vs qmc vs gps）
      └─ M6: 三档预算落到 OOS 折数
  ↓
P2  LLM 语义探索（PoC，护栏不放松）
      ├─ M7: tune/llm_suggest.py + study.enqueue_trial
      └─ M8: LLM 组 vs 纯 TPE 组 trial 数对比 + OOS 不退化校验
```

**里程碑完成判据**：每个 M 都有可复现的对照证据（OOS 衰减案例 / 收敛曲线 / trial 数对比），不是"代码写完了"。

---

## 7. 风险与护栏

| 风险 | 应对 |
|---|---|
| OOS 反复 peek（用 OOS 调参） | 保留段一次性封印（复用 §6.1 `HoldoutSealedError`）；OOS 只记录不进目标 |
| LLM 给"讲道理但过拟合"的参数 | P0 的 OOS 判定线兜底；衰减超阈值即 RESEARCH_ONLY |
| 换采样器后不可比 | 所有采样器共享同一 objective（composite）+ 同一 OOS 验证 + 同一 seed |
| 全样本回测太慢拖垮 P0/P1 | P1 M6 三档预算 + 粗筛精调两阶段压低墙钟 |

---

*文档结束。本文为调优引擎升级设计，不含业务代码，未修改任何现有文件；实施时按 §6 里程碑逐项落地。*
