# 妙算（MiaoSuan）测试验证报告

- **报告人**：QA 工程师「严过关」（software-qa-engineer）
- **生成时间**：2026-09-16
- **被测项目**：D:\backup\BaoBao\PythonProgram\miaosuan（妙算 v0.1.0）
- **测试目标**：搭建运行环境、运行完整测试套件、确认应用能否正常运行

---

## 1. 环境信息

| 项目 | 值 |
|------|----|
| 托管 Python | C:\Users\Administrator\.workbuddy\binaries\python\versions\3.13.12\python.exe（实际 3.13.14） |
| 虚拟环境 | D:\backup\BaoBao\PythonProgram\miaosuan\.venv（venv，由上述托管 Python 创建） |
| 操作系统 | Windows（win32） |
| pytest | 9.1.1 |
| pytest-cov | 7.1.0 |
| hypothesis | 6.168.0 |
| numpy | 2.5.3 |
| pandas | 2.3.3 |
| pyarrow | 25.0.1 |
| scipy | 1.18.1 |
| pydantic | 2.13.5 |
| pyyaml | 6.0.3 |
| jinja2 | 3.1.6 |
| typer | 0.27.2 |
| torch | **未安装**（符合预期：parity 对拍使用冻结 `.npz` 基准 + 自动注入 torch 桩，AM Oracle 仓库已存在且可用） |
| optuna / TA-Lib | 未安装（可选依赖，默认测试套件不依赖，详见 §3） |

> 说明：torch 缺失时，`tests/parity/conftest.py` 会自动向 `sys.modules` 注入最小 torch 桩，仅用于「读取 AM 词表 token 名称」这一导入期动作，不影响任何被断言的数值；M5 / E2E / features / ops 等数值对拍直接读取 `tests/fixtures/` 下的冻结 `.npz`，无需 torch。

---

## 2. 环境搭建与依赖安装

### 实际情况（与任务简报的差异）

任务简报标注「项目当前没有 .venv，依赖未安装」。经核查，**项目根目录已存在一个完整、可用的 `.venv`**，且其 `pyvenv.cfg` 显示它是由上述托管 Python 3.13.12 创建的，已包含：

- 全部运行时核心依赖（numpy / pandas / pyarrow / scipy / pydantic / pyyaml / jinja2 / typer）；
- 全部 dev 依赖（pytest / pytest-cov / hypothesis / ruff / mypy / pip-tools / pre-commit）；
- `miaosuan` 的**可编辑安装**（editable install，`_editable_impl_miaosuan.pth` 已注册）。

### 采取的 QA 动作

作为 QA，未盲目重建 venv（重建会破坏已就绪环境，且 `tune` extra 会引入需本地 C 库的 TA-Lib，存在安装失败风险）。改为**逐项验证环境完整性**：

```bash
# 验证全部关键包可导入并取得版本（脚本节选）
.venv/Scripts/python.exe -c "import numpy, pandas, pyarrow, scipy, pydantic,
    yaml, jinja2, typer, pytest, hypothesis, miaosuan; print('all importable')"
```

验证结果：12 个关键模块全部 `OK`，`miaosuan` 可导入且版本为 `0.1.0`。**结论：运行环境已具备，无需重新安装依赖。**

### 安装命令（如未来需在干净环境复现，推荐顺序）

```bash
# 1) 创建 venv（用托管 Python）
C:\Users\Administrator\.workbuddy\binaries\python\versions\3.13.12\python.exe -m venv .venv

# 2) 安装运行时不依赖 torch/TA-Lib/optuna 的默认集合
.venv\Scripts\pip.exe install -e ".[dev]"        # 运行时 + dev（不含 oracle/torch/tune）
# 或回退方案：
.venv\Scripts\pip.exe install numpy pandas pyarrow scipy pydantic pyyaml jinja2 typer pytest pytest-cov hypothesis
.venv\Scripts\pip.exe install -e .
```

> 本机已满足上述全部条件，故未执行安装（pip 无操作即为成功）。

---

## 3. 完整测试结果汇总

**命令**：`.venv\Scripts\python.exe -m pytest tests -v`（项目根目录执行，等价于 `make test`，覆盖 `tests/` 与 `tests/parity`）

| 指标 | 数值 |
|------|------|
| 收集用例总数（collected） | **919** |
| 通过（passed） | **915** |
| 跳过（skipped） | **3** |
| 失败（failed） | **1** |
| 错误（error） | **0** |
| 告警（warnings） | 3（均为 `RuntimeWarning: invalid value encountered`，来自 `core/ops.py` 的 NaN/除零处理，符合预期并已 filter） |
| 耗时 | **69.08s** |

**通过率**：915 / 919 ≈ **99.67%**（不计 3 个设计性跳过则为 99.89%）

### 覆盖范围

- `tests/parity/*`：与冻结 AlphaMaster 基准的差分对拍（vm / signal / backtest / evaluator / features / ops / vocab / e2e_xauusd / token 顺序 / feature 因果性）——全部通过（含 torch 桩代管与 `.npz` 基准）。
- 核心模块：vocab、ops、config、errors、logging、ir（codec/schema/provenance）、market_profiles、search（ga/islands/mine/rpn/budget）、shenji（export/fidelity/kernel/lint/magic/realtime）、gate（cost_curve/holdout/multiple_testing/verdict）、adapters、data_loader/panel/split、pipeline、report（equity/metrics）、cli、webui、core_zero_touch、no_placeholder_left 等。

---

## 4. 失败 / 错误用例清单与根因归类

### 4.1 失败用例（1 项）

| 测试 | 文件:行 | 类型 |
|------|---------|------|
| `test_os_environ_only_in_config_and_cli` | tests/test_dependency_direction_adapters.py:102 | **代码性失败（架构合规违规）** |

**断言信息**：
```
AssertionError: 读取环境变量只允许在 config.py / cli.py：
data\acquisition.py:139 os.environ,
data\acquisition.py:195 os.environ,
data\acquisition.py:196 os.environ,
data\acquisition.py:293 os.environ
```

**根因**：该项目有一条明确的架构约束——「读取环境变量只允许在 `config.py` / `cli.py`」。该测试通过 AST 静态扫描 `src/` 下所有源文件，发现 `src/miaosuan/data/acquisition.py` 在以下位置**直接**调用 `os.environ.get(...)` 读取配置：

- 第 139 行：`_ENV_SHENJI_DB`（妙算数据库路径）
- 第 195 行：`_ENV_DATA_API_URL`（数据 API 地址）
- 第 196 行：`_ENV_TIMEOUT`（超时）
- 第 293 行：`_ENV_CACHE_DIR`（缓存目录）

**归类**：**代码性失败（code）**，非环境性。原因不依赖 Python 版本、依赖或网络——在任何机器上都会失败。它是一个**架构合规性静态检查**失败：源码未遵守项目自身的「环境变量集中管理」约定。该检查为静态 AST 分析，不影响运行时行为（应用可正常导入与执行），但属于应修复的代码质量问题。

**建议修复方向**：将 `data/acquisition.py` 中的 4 处 `os.environ.get(...)` 改为从 `AppConfig` / `config.py` 注入相应配置项（或经由 `config.py` 统一读取后下发），使环境变量访问收敛到允许的模块内。

### 4.2 跳过用例（3 项，设计为跳过，非失败）

| 测试 | 文件:行 | 跳过原因 |
|------|---------|----------|
| `test_feature_causality.py` 中 OBV_SLOPE 相关 | :120 | OBV_SLOPE 记忆无界（带状态 / cumsum 平移），不参与方向 2 |
| `test_feature_causality.py` 中 AD_LINE_SLOPE 相关 | :120 | AD_LINE_SLOPE 记忆无界，不参与方向 2 |
| `test_feature_causality.py` 中 SUPERTREND_DIR 相关 | :120 | SUPERTREND_DIR 记忆无界，不参与方向 2 |

这 3 项属**预期内的设计性跳过**，已在测试内以 `pytest.skip(...)` 显式标注，不计为失败。

### 4.3 错误（error）：无

---

## 5. 冒烟复核（独立运行）

**命令**：`.venv\Scripts\python.exe -m pytest tests/test_smoke.py -v`

**结果**：**6 passed in 0.52s**（退出码 0）

**关键断言复核（已通过）**：
- 包可导入：`miaosuan.__version__` 与 `TOOL_NAME == "miaosuan"` ✅
- `VOCAB_VERSION == "v9217a2c0d91a"` ✅（独立 `python -c` 复核一致）
- `FORMULA_VOCAB.feature_count == 65` ✅
- `len(FORMULA_VOCAB.operator_names) == 62` ✅
- `FORMULA_VOCAB.size == 127` ✅
- 配置默认值 / 错误体系 / JSON 行日志 run_id 等均通过 ✅

---

## 6. 结论

**应用能否正常运行：可以（功能层面完全正常，部分通过）。**

- 完整测试套件 **919** 项中 **915 通过、3 设计性跳过、1 代码性架构合规失败、0 错误**；
- 所有**功能性与对拍（parity）测试均通过**，torch 缺失由桩与冻结 `.npz` 基准正确代管，未影响结果；
- 冒烟测试 6/6 通过，`VOCAB_VERSION == v9217a2c0d91a`（feature 65 / operator 62 / vocab 127）已确认；
- 唯一失败为 `test_os_environ_only_in_config_and_cli`：属**代码层架构违规**（`data/acquisition.py` 在 config.py/cli.py 之外直接读取环境变量），**非环境缺失、非运行时崩溃**，不影响应用实际运行，建议按 §4.1 方向修复。

**判定**：
- ✅ 运行环境：可用（Python 3.13.14 + 完整依赖 + 可编辑安装）。
- ✅ 应用功能：正常，可通过完整测试与对拍验证。
- ⚠️ 待修复（代码质量，非阻塞）：1 处环境变量读取越权，需收敛至 `config.py` / `cli.py`。

> 说明：本报告对应的原始日志为 `tests_full.log`（完整套件）与 `smoke.log`（冒烟），已生成于项目根目录备查。
