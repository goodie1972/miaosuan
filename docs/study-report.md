# 妙算（MiaoSuan）应用研究报告

> 研究性质：静态代码研读 + 架构分析（**未修改任何应用代码**）。
> 研究依据：实际通读 `README.md`、`pyproject.toml`、`Makefile`、包初始化、`config.py`、`cli.py`、`core/vm.py`、`core/vocab.py`、`pipeline.py`、`adapters/shenji/generator.py`、`data/loader.py`、`market/profiles.py`，并对 `core/` 与 `adapters/` 的 import 做抽查验证。
> 报告日期：基于代码当前状态（T01–T07 完成，AGPL-3.0）。
> 证据格式：所有结论均标注 `文件:行号` 可追溯出处。

---

## 1. 项目定位与一句话价值

妙算是一套**离线量化研究与策略生产工具**，定位为「去 Torch 的声明式因子挖掘 + 妙算（Shenji）策略调优/导出引擎」（`pyproject.toml:22`、`README.md:1-3`）。

**一句话价值**：用纯 NumPy 在「可复现、轻量（~100MB）、零 GPU 依赖」的前提下，把 RPN 遗传编程因子搜索与 Optuna 参数寻优统一到同一份 `StrategySpec` IR，并把可信因子一键导出为可上线的妙算策略。

两种核心模式（对齐 `README.md:16-24`）：
- **模式 A（因子挖掘）**：行情 Panel → RPN-GA 结构搜索 → 可信因子公式 → 导出妙算策略。
- **模式 B（参数寻优）**：已有妙算策略 → Optuna TPE + 事件驱动回测 → 外提最优参数 → 回写新版本。
- 两模式经 `StrategySpec`（`ir/schema.py`）打通，形成 `mine → export → tune → export v2 …` 闭环。

---

## 2. 模块划分与职责

依赖方向（自上而下，禁止反向，`__init__.py:12-18`）：

```
cli → pipeline → {search, tune, gate, ir, report}
                         ↓
                       core（纯 numpy，无 IO / 无环境变量 / 无平台知识）
adapters ← pipeline（core 不得 import adapters）
```

| 层 / 模块 | 关键文件 | 职责 |
|---|---|---|
| **顶层入口** | `cli.py` | 唯一允许做文件 IO / 读环境变量的层；6 个子命令 `mine/export/verify/report/backtest/ui` + `tune`（`cli.py:60-64`、`cli.py:148-575`） |
| **配置树** | `config.py` | 单一配置源（`AppConfig` + 子 config 全部 frozen dataclass）；依赖注入；唯一 `from_env` 读环境变量入口（`config.py:196-275`） |
| **编排层** | `pipeline.py` | 「模式 A」唯一编排入口 `run_mine`：数据→切分→搜索→门禁→`StrategySpec`；不含 IO，可独立测试（`pipeline.py:194-256`） |
| **纯 numpy 内核** `core/` | `registry.py` `features.py` `ops.py` `vocab.py` `vm.py` `evaluator.py` `backtest.py` `signal.py` `ports.py` | 65 维特征注册、62 算子注册、确定性 `VOCAB_VERSION`、StackVM 公式求值、夏普/衰减评估、tanh 仓位回测；**铁律：不 import torch / 不 IO / 不读 env / 不反向依赖上层**（`__init__.py:10`、`features.py:22`、`ops.py:19`） |
| **适配器** `adapters/shenji/` | `base.py` `generator.py` `kernel.py` `lint.py` `magic_registry.py` `contract.py` `realtime.py` | 妙算策略双向适配层：`StrategySpec → .py`（正向生成，Jinja2 模板）+ `.py →` 数值保真度（反向抽取）；`install_stub_modules` 离线桩机制（`base.py:320-361`） |
| **数据层** `data/` | `loader.py` `panel.py` `split.py` `acquisition.py` `fingerprint.py` `sources/parquet_mt.py` | Parquet/CSV → `Panel`；按 `{symbol}_{timeframe}` 推断；时间戳单位修复；两段切分 + hold-out 封印（`loader.py:69-223`） |
| **市场画像** `market/` | `profiles.py` `cost.py` | 冻结 `FrozenMarketProfile`（14 必填 + 1 可选）；4 个内置实例；新增市场零改 `core/`（`profiles.py:78-282`） |
| **门禁** `gate/` | `holdout.py` `cost_curve.py` `multiple_testing.py` `verdict.py` | hold-out 封印台账、成本敏感度曲线、多重检验校正、门禁裁决（`pipeline.py:17` 不消费封印） |
| **IR** `ir/` | `schema.py` `codec.py` `provenance.py` | `StrategySpec` 数据 schema、`read/write_spec` 编解码、`provenance` 溯源（git_sha/指纹/种子/配置快照） |
| **报告** `report/` | `metrics.py` `equity.py` `persist.py` | 绩效指标、事件驱动回测（`run_full_backtest`，`cli.py:55` 引用）、持久化 |
| **搜索器** `search/` | `ga.py` `rpn.py` `mine.py` `islands.py` `budget.py` | RPN-GA 遗传编程（岛屿模型）、预算/早停 |
| **调优器** `tune/` | `engine.py` | Optuna TPE（**lazy import**，仅模式 B 需要，`cli.py:56`） |
| **Web UI** `webui/` | `server.py` | FastAPI + uvicorn 仪表盘（`cli.py:560-575`） |

---

## 3. 依赖方向铁律及其在代码中的体现

### 3.1 铁律原文（README §3、§8；`__init__.py:12-18`）
1. `core/` **不得** `import torch`；
2. `core/` **不得** 读环境变量、**不得** 做文件 IO（配置与数据一律参数注入）；
3. `core/` **不得** import `adapters/`、`tune/`、`cli.py`；
4. 配置单一来源：`config.py` 的 dataclass 树，依赖注入向下传递（`README.md:233`）；
5. 环境变量**只允许**在 `config.py` / `cli.py` 读取（`test_dependency_direction_adapters.py:92-102`）。

### 3.2 抽查 import 验证（证据）

**core/ 无 torch、无 IO、无 env、无反向依赖**
- 对 `core/*.py` 全文检索 `torch|import os|open(|os.environ`，命中的全部是**注释/文档字符串**中关于「去 torch 化移植」的说明（`features.py:6-34`、`ops.py:8-32`、`vm.py:8-31`、`registry.py:6-28`、`background` 等），**没有任何真正的 `import torch` 语句**（`core/features.py`、`core/ops.py`、`core/vm.py`、`core/registry.py`、`core/evaluator.py`、`core/signal.py`、`core/backtest.py`）。
- `core/` 内部只允许相对 import 兄弟模块，例如 `vm.py:41-42` 仅 `from .ops import OPS_CONFIG` / `from .vocab import FORMULA_VOCAB`；`backtest.py:35` `from .signal import ...`（`Grep` 对 `core/` 的 `import|from` 结果仅此一条业务 import，无 `adapters/tune/cli`）。
- `features.py:22`、`ops.py:19` 在模块 docstring 中明文声明「**禁止** import torch、禁止文件 IO、禁止读取环境变量（CI 的 …）」。

**CI 把铁律做成了可执行断言（而非仅靠自觉）**
- `tests/test_dependency_direction.py`：用 **AST 静态扫描** `src/miaosuan/core/*.py`，逐条禁 `import torch`、`import adapters/tune/cli`、`os.environ/getenv/putenv`、`open`/`Path.read_text` 等文件 IO（避免把注释里的 "torch" 误判），见 `test_dependency_direction.py:61-124`。
- `tests/test_dependency_direction_adapters.py`：断言 `adapters` 不得 import `cli/tune/search/gate`（`test_dependency_direction_adapters.py:54-117`）；`AppConfig.from_env` **只允许**在 `cli.py` 调用（`:75-89`）；`os.environ/getenv/putenv` **只允许**在 `config.py`/`cli.py`（`:92-102`）；`adapters` 永不 import torch（`:61-68`）。
- `tests/test_core_zero_touch.py`：断言新增市场画像不触碰 `core/`（与 `profiles.py:23` 声明一致）。
- `tests/test_no_placeholder_left.py`：断言无遗留占位 `TODO`/占位桩。

---

## 4. 关键数据流：`mine → export → verify → report → backtest → ui`

六个 CLI 命令（`cli.py` 的 `@app.command()`）各自职责与代码落点：

| 命令 | 入口行 | 它做什么 | 关键调用 |
|---|---|---|---|
| **mine** | `cli.py:148-229` | 数据→封印 hold-out→RPN-GA 搜索→门禁→`StrategySpec` JSON。`holdout_sharpe` 故意留 `None`（诚实不消费封印，`pipeline.py:167`） | `load(data)` → `run_mine(...)`（`pipeline.py:194`）→ `write_spec(out, spec)` |
| **export** | `cli.py:233-284` | `StrategySpec` → 妙算 `.py`（Jinja2 渲染）+ 自动分配 magic + 静态检查 lint。不写盘的渲染逻辑在 `generator.py`，写盘由 CLI 负责 | `ShenjiPort(...).compile(spec)`（`generator.py:357`/`base.py:252`） |
| **verify** | `cli.py:288-327` | 对导出 `.py` 做 repaint lint；若带 `--data` 则额外做**数值保真度回归**：导入生成的 `compute_factor`，与 `StackVM` 原生求值逐位比对，阈值 `1e-3`（`cli.py:344-366`） | `lint_source(source)` + `_fidelity_error(...)`（StackVM 比对） |
| **report** | `cli.py:371-404` | 打印 spec 的**证据 + 溯源**摘要（val_score、DSR、hold-out、成本敏感度、git_sha、数据指纹、种子），不产出文件 | `read_spec` → 输出 `evidence`/`provenance` |
| **backtest** | `cli.py:408-487` | 全样本事件驱动回测：tokens + 行情 + 画像 → 逐 bar 资金曲线/滚动夏普/交易统计。**口径提醒**：Sharpe 由逐 bar 净收益算，与 `val_score`（搜索适应度）不是一回事（`cli.py:418-485`） | `run_full_backtest(...)`（`report/equity.py`，`cli.py:55`） |
| **ui** | `cli.py:560-575` | 启动本地 FastAPI + uvicorn 仪表盘（仅本机 127.0.0.1） | `uvicorn.run(create_app(), ...)`（`webui/server.py`） |

> 模式 B 另有 `tune` 命令（`cli.py:491-556`）：对已导出策略的语义参数做 Optuna TPE 寻优，依赖 `tune/engine.py`（**lazy import**，见 `cli.py:56`、§7）。

---

## 5. 如何运行

### 5.1 Makefile 目标（统一本地与 CI 入口，`Makefile:29-84`）
| 目标 | 命令 | 作用 |
|---|---|---|
| 建环境 | `make env [PYTHON_BOOT=python3.11]` | 建 `.venv` + 装 `.[dev]` + pre-commit 钩子（`Makefile:43-48`）；本机用 3.13 对齐（`Makefile:22`） |
| 自检 | `make smoke` | 跑 `tests/test_smoke.py`：import 妙算包 + vocab 自检（`Makefile:51-52`） |
| 全测 | `make test` | `pytest tests`（含 `tests/parity` 对拍，`Makefile:55-56`） |
| 对拍 | `make parity` | `pytest tests/parity`（冻结基准差分，`Makefile:58-59`） |
| 静态 | `make lint` | `ruff check src tests` + `mypy`（`Makefile:67-69`） |
| 格式化 | `make format` | `ruff format`（`Makefile:71-72`） |
| 重建基准 | `make fixtures` | 用装了真实 torch 的 `ORACLE_PYTHON` 重新生成冻结基准（`Makefile:75-80`） |

Windows 自动用 `.venv/Scripts/python.exe`，POSIX 用 `.venv/bin/python`（`Makefile:12-20`）。

### 5.2 CLI 命令示例（README §6、§5）
```bash
# 模式 A 闭环
miaosuan mine   --data XAUUSD_H1.parquet --budget standard --out artifacts/spec.json
miaosuan export --spec artifacts/spec.json --out-dir shenji-strategies
miaosuan verify --file shenji-strategies/20260910_xauusd_miaosuan_v1.py --data XAUUSD_H1.parquet
miaosuan report --spec artifacts/spec.json
miaosuan backtest --spec artifacts/spec.json --data XAUUSD_H1.parquet
miaosuan ui     --port 8686

# 模式 B 参数寻优（可选，需 tune 依赖）
miaosuan tune --spec artifacts/spec.json --data XAUUSD_H1.parquet --n-trials 50 --metric sharpe
```
> 预算档位 `quick/standard/deep` 由 `config.BudgetConfig` 预置（`config.py:123-133`），`mine` 命令在 `cli.py:166-167` 校验合法档位。

---

## 6. 测试布局与对拍机制

### 6.1 测试总量与分层
- `tests/` 下共 **53 个测试文件**（36 个功能测试 + `tests/parity/` 下 17 个对拍文件，含 `conftest.py`/`feature_cases.py`/`m5_cases.py`/`ops_cases.py` 与 13 个 `test_*.py`），与「约 50 个测试文件」的梳理一致。
- 功能测试按模块覆盖：`test_core_zero_touch`、`test_data_loader/panel/split`、`test_dependency_direction(_adapters)`、`test_gate_*`(4)、`test_ir_*`(3)、`test_market_profiles`、`test_pipeline`、`test_report_*`(2)、`test_search_*`(5)、`test_shenji_*`(6)、`test_cli`、`test_webui`、`test_smoke`、`test_adapters_base`、`test_no_placeholder_left`。

### 6.2 对拍（parity）机制 —— 与冻结 AlphaMaster（torch）差分
- **目的**：妙算（numpy）须与冻结的 AlphaMaster（原 torch 实现）在数值上逐位/逐字符一致（`README.md:12`、`tests/parity/conftest.py:6-8`）。
- **两级 Oracle**（`tests/parity/conftest.py:10-27`）：
  1. **冻结快照（始终可用）**：`tests/fixtures/am_vocab_snapshot.json` 记录了 AM 词表版本串与有序 token 名，离线、无需 torch（`conftest.py:13-15`、`57-61`）。实测值：`vocab_version=v9217a2c0d91a`、`feature_count=65`、`operator_count=62`、`vocab_size=127`。
  2. **活体对拍（可选，装了 torch 时启用）**：直接 `import model_core.vocab` 读真实 AM 模块比对（`conftest.py:17-18`、`122-151`）。
- **torch 桩机制**（`conftest.py:64-116`）：AM 的 `model_core/{registry,features,ops}.py` 仅在模块顶层 `import torch`（用于类型注解，导入期不执行数值运算）。当本机无 torch 时，`_ensure_torch()` 向 `sys.modules` 注入一个最小 `_TorchStubModule`（万能 `_AnyStub`，任何属性/调用/下标都返回自身），仅用于「读名称」这一动作，**不影响任何被断言的值**；若已装真实 torch 则优先用真包（`conftest.py:90-116`、`_torch_available`）。
- **vocab 恒等性断言**：`test_vocab_identity.py` 在每次 CI 断言 `VOCAB_VERSION` 恒等（`README.md:200`）；`test_frozen_token_order.py` 守 token 顺序；改一个算子或顺序即测试变红。
- **数值基准**：`m5_baseline.npz/json`、`e2e_xauusd_baseline.npz` 由 `make fixtures` 用真实 torch 生成（`conftest.py:50-55`、`188-219`），缺失则对应测试 `pytest.skip`。

---

## 7. 已知差异（README §9）

| 项 | 架构目标 | 本机实际情况 | 处理 |
|---|---|---|---|
| **Python 版本** | 3.11（`requires-python = ">=3.11"`，`pyproject.toml:24`） | 本机仅 3.13.12 / 3.14.3，无 3.11 | 降级用 **3.13** 开发验证（`README.md:239`）；CI 矩阵仍按 3.11 保证 |
| **numpy 版本** | `>=1.26,<2.1` | 3.13 需 `numpy>=2.1`，无 cp313 wheel | `requirements.lock` 按 3.13 可装版本锁定（实测 numpy 2.5.x）；mypy `python_version=3.13`、ruff `target-version=py311`（`pyproject.toml:101-121`） |
| **算子数量** | 66 个 | 实际 **62 个**（44 基础 + 3 跨截面 + 8 Task3.3 + 7 Task3.4） | 以实测为准；`VOCAB_VERSION` 用 62 个算子派生（`README.md:241`，与 `am_vocab_snapshot.json` 一致） |
| **特征数量** | 65 维 | 实测 65 维 | 一致（`features.py:1169` `FEATURE_REGISTRY`，冻结快照 `feature_count=65`） |

---

## 8. 给 QA 的「测试范围与推荐运行方式」

### 8.1 推荐运行方式（默认即可，零额外依赖）
```bash
# 推荐：一条命令跑全量（功能测试 + parity 对拍）
make test

# 或等价
.venv/Scripts/python.exe -m pytest tests -q
```
- **torch 可缺省**：`tests/parity` 的 torch 依赖由 `conftest.py` 的 `install_stub_modules`/`_TorchStubModule`（§6.2）与缺失基准的 `pytest.skip`（§6.2）兜底，**未装 torch 也能跑全量**；活体对拍与数值基准仅在装了真实 torch 且 `tests/fixtures/*.npz` 存在时才启用。
- **optuna / TA-Lib 为可选**：仅在模式 B `tune` 命令路径与对应测试中 lazy import（`cli.py:56`、`tune/engine.py`）。**它们未被 `tests/` 顶层 import 即加载**——`make test` 不会因缺 `optuna`/`TA-Lib` 而失败。若需验证模式 B，再 `pip install -e ".[tune]"` 后单独跑 `test_search_*`/`tune` 相关用例即可。
- **oracle（torch）依赖**仅列在 `pyproject.toml:53` 的 `oracle` extra，仅供 `make fixtures` 重建冻结基准，不参与日常测试。

### 8.2 建议 QA 重点覆盖范围
1. **依赖方向铁律回归**（高优先级，防架构腐化）：`test_dependency_direction.py`、`test_dependency_direction_adapters.py`、`test_core_zero_touch.py` —— 任何新增 `import torch`、新增 env 读取、新增 `adapters→cli/tune` 引用都会立刻变红。
2. **vocab / 算子 / 特征 恒等 & 对拍**：`tests/parity/test_vocab_identity.py`、`test_frozen_token_order.py`、`test_ops_parity.py`、`test_feature_parity.py`、`test_vm_parity.py`、`test_backtest_parity.py`、`test_evaluator_parity.py`、`test_signal_parity.py` —— 守护「numpy 移植与冻结 AM 逐位一致」。
3. **端到端闭环**：`test_cli.py`（mine→export→verify→report→backtest 串跑）、`test_pipeline.py`、`test_shenji_export.py`、`test_shenji_fidelity.py`（数值保真度回归，`cli.py:344` 阈值 1e-3）。
4. **门禁与诚实性**：`test_gate_*` 验证 hold-out 封印、成本敏感度、多重检验、门禁裁决；`pipeline.py:167` 确保 `holdout_sharpe=None` 不消费封印。
5. **配置/画像/数据**：`test_config` 相关、`test_market_profiles.py`（零改主干）、`test_data_loader.py`/`test_data_panel.py`/`test_data_split.py`。

### 8.3 QA 注意事项
- 数值基准文件（`m5_baseline.npz`、`e2e_xauusd_baseline.npz` 等）若缺失，相关 parity 用例会 `skip` 而非失败，QA 不应误判为通过。
- `make lint`（ruff + mypy strict）建议纳入门禁；mypy 设 `python_version=3.13` 是为匹配本机 numpy 2.5 的 PEP 695 `.pyi`，语法层 3.11 兼容由 ruff（`target-version=py311`）与 CI 3.11 矩阵保证（`pyproject.toml:99-121`）。
- `backtest` 的 Sharpe/Sortino 与 `spec.evidence.val_score`（搜索适应度）**口径不同**，QA 做绩效断言时勿混淆（`cli.py:418-485` 有明确提醒）。

---

## 9. 结论摘要

妙算是一套**架构纪律极强**的离线量化工具：以 `core/`（纯 numpy、零副作用）为内核，通过统一的 `StrategySpec` IR 串联「挖掘→导出→校验→报告→回测→调优」闭环；其依赖方向铁律（无 torch / 无 IO / 无 env / 无反向依赖）不仅写进文档，更被 `tests/test_dependency_direction*.py` 等以 **AST 静态扫描**变成 CI 可执行断言，配合 `tests/parity` 的「冻结 AM（torch）差分对拍 + torch 桩兜底」机制，在「轻量可复现、零 GPU 依赖」的前提下保证数值逐位可信。对 QA 而言，`make test` 一条命令即可覆盖全量（含对拍），torch/optuna/TA-Lib 均为可选，不影响主流测试运行。
