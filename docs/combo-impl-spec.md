# 多因子组合（combine）实现规格

> 主理人写给工程师的实施规格。设计来源：架构师研究（2026-09-26，四问答复见下）。
> 本文件是唯一权威规格；IM 消息若与此冲突，以本文件为准。

## 0. 背景与目的

XAUUSD 四周期（H1/M15/M30/H4）deep 预算挖掘**全部被 `dsr<0.5` 门禁 BLOCKED**
（H1 最佳 DSR 0.3714，其余 ≤0.11）。方向③：把多个弱因子组合成组合信号，
过**同一套** walk-forward + DSR 门禁，看能否汇聚出统计显著的信号。

现状代码库无任何 combo/ensemble 能力（已 grep 确认）。

## 1. 主理人拍板的决策（不可更改）

1. **同周期内组合**（首版）。跨周期 bar 轴不同，留二期。
2. 权重：**默认等权**（零自由参数）；`--method ic` 可选（IC/val_score 确定性加权）。
   **禁止 GA 搜权重**。
3. **n_trials 必须累加**：`n_trials_combo = Σ(各成员「来源运行」的 n_trials，按运行去重)
   + 权重搜索次数（等权=0)`。同一运行的两个候选只计一次该运行的 n_trials。
   理由：DSR 零假设是 N 次独立试验的最大 Sharpe；不校正会把 BLOCKED 因子洗成 DEPLOYABLE。
   成员 n_trials 来源：候选落盘 JSON 的 `verdict_snapshot.n_trials`（即该次运行 Evidence.n_trials）。
4. **组合层面**：因子（z-score）层加权 → 符号对齐（按 IC 统一方向）→ 加权和 →
   因果滚动重标定（roll_window=500，无前视）→ **单次** tanh → **单次** neutral_band=0.05
   → long_short。**禁止仓位层平均**（各弱因子小信号在合并前就被自己的中性带清零）。
5. **只在开发区评估**：`make_split(..., seal=False)` 重建切分（split.py:253 明文路径，
   不触发 E-HOLDOUT-SEALED），`panel.slice_view(0, val_end)`，与 mine() 同款结构。
   hold-out 结构性读不到；一次性 hold-out 验收**后置不做**。
6. `ComboPayload` 在 spec 1.x 内以新 kind 兼容新增，**不 bump SPEC_VERSION 2.0**。
7. 门禁与单因子**完全同款**：evaluate_cost_curve → deflated_sharpe_ratio（n_trials 用 Σ 累加值）
   → evaluate_verdict（mine.py `_gate_candidate` 同序同参）。

## 2. 任务0（先做，小修）：候选落盘 JSON 补字段

commit a180440 的 `--dump-candidates` 缺以下字段（combine 的校验输入）：
在 JSON **顶层**补：`vocab_version`、`data_fingerprint`、
`semantics{timeframe, position_fn, neutral_band, long_short}`（全体候选共享，顶层合理）、
`n_trials`（本次运行总评估数，顶层一份即可；候选内 verdict_snapshot 已有的保留）。
补测试断言四字段存在且与本次运行一致。单独小 commit。

## 3. 实现结构（T1→T4，T1/T2 可并行）

| 任务 | 文件 | 内容 |
|---|---|---|
| T1 | `src/miaosuan/core/combo.py`（新） | 纯 numpy，不依赖 ir/不读 env/不做 IO。函数：`sign_align(factors, ic) -> np.ndarray`；`combine_factors(factors: list[np.ndarray], weights: np.ndarray, *, roll_window: int = 500) -> np.ndarray`（加权和→因果滚动重标）；`combo_to_position(agg, *, neutral_band, long_only) -> np.ndarray` |
| T2 | `src/miaosuan/ir/schema.py`（改） | `ComboPayload(kind="combo", members: tuple[tuple[int,...],...], weights: tuple[float,...], vocab_version: str)`；`StrategySpec.from_dict` 增 kind 分派；spec_id/codec 相应支持 |
| T3 | `src/miaosuan/search/combo.py`（新）+ `src/miaosuan/pipeline.py`（改） | `@dataclass ComboCandidate(members, weights, val_score, dsr, sharpe_2x, wf_win_rate, verdict, verdict_snapshot, n_trials)`；`combine(...)` 做组合评估+门禁（§1.7）；`pipeline.run_combine(candidates_paths, *, panel, config, method, seed, git_sha) -> ComboOutcome`：seal=False 重建切分→dev 区算 features/target_ret→成员校验→组合→门禁→组装 ComboStrategySpec |
| T4 | `src/miaosuan/cli.py`（改） | `combine` 子命令（§4）；report/backtest **暂不**扩展 combo payload（后置），但 combine 自身要打印 verdict/DSR 并写产物 |

**成员校验（combine 入口，不满足即 `_die`）**：同 timeframe、同 vocab_version、
同 data_fingerprint、timeframe 来源非「配置默认」时才可信（任务B 已修）。

## 4. CLI

```
miaosuan combine \
  --candidates a.candidates.json,b.candidates.json \
  --data <同周期数据文件> \
  --method equal|ic \
  --seed 20260910 \
  --out artifacts/combo_h1.json
```
- 输入为任务A 落盘的 candidates JSON（**不是** spec 文件——里面有 n_trials/dsr/verdict）。
- 取每个文件的 rank0..k（默认 rank0..2，`--top-per-file n` 可调），跨文件去重相同 tokens。
- 产物：ComboStrategySpec JSON（含 members/weights/evidence.n_trials=Σ/verdict/provenance）。

## 5. 测试与铁律

- `tests/test_combo.py`：等权/IC 加权数值、符号对齐、n_trials 按运行去重累加、
  同周期校验拒绝（timeframe/vocab/fingerprint 任一不一致）、seal=False 不触发封印、
  门禁快照与单因子口径一致、CLI 集成（构造小 candidates 文件端到端）。
- 全量测试：`.venv\Scripts\python.exe -m pytest tests -q --basetemp .pytest_r11 -p no:cacheprovider`
  （basetemp 必须全新不存在 + 必须禁 cacheprovider，缺一产生几十个假 ERROR）。
  当前基线 **1013 passed / 3 skipped / 0 failed**，只增不减。
- ruff/mypy 过改动文件。禁止改 `core/` 既有语义、禁止动 `data/split.py` 封印逻辑、
  env 只允许在 config.py/cli.py 边界读取。
- 完成后 commit（可 T1/T2 一条、T3/T4 一条），报告：commit hash、CLI 用法实测输出
  （verdict/DSR/n_trials 三值）、测试通过数。

## 6. 后置（本规格不做）

- report/backtest/export 接受 ComboPayload；神机导出（架构师 T5）；hold-out 一次性验收；
  跨周期组合；受约束权重搜索（备选 P2）。
