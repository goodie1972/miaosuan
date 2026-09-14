# 妙算（MiaoSuan）

> 定位：把量化因子挖掘系统 **AlphaMaster(AM)** 重构为「去 torch、可复现、插件化」的新程序。
> 一句话：**一套纯 numpy 的声明式因子语言 + 一个遗传编程搜索器 + 一个 神机 策略双向适配层。**

---

## 1. 妙算是什么

妙算是一套**离线**的量化研究与策略生产工具，聚焦两件事：

| 模式 | 输入 | 输出 | 引擎 |
|---|---|---|---|
| **模式 A：因子挖掘** | 行情数据（Panel） | 一份可信的因子公式 → 可导出的 神机 策略 | RPN-GA 结构搜索 |
| **模式 B：参数寻优** | 已有 神机 策略 | 外提参数后的最优参数集 → 回写策略新版本 | Optuna TPE + 事件驱动回测 |

两种模式通过**统一的产物 IR（`StrategySpec`）** 打通：模式 A 的导出物可直接作为模式 B 的输入，
形成 `mine → export → tune → export v2 → tune …` 的闭环。

## 2. 架构一览（Hexagonal-lite）

```
CLI  ──►  Pipeline  ──►  {search(GA) | tune(Optuna)}  ──►  core(纯 numpy)
                          │                                  ▲
                    data/ + market/（可插拔）             ir/ + gate/ + report/
                          │
                    adapters/shenji（正向生成 / 反向抽取）
```

**依赖方向铁律（CI 强制，架构 §1.2 / §9.7）**：

- `core/` **不得** `import torch`；
- `core/` **不得** 读环境变量、**不得** 做文件 IO（配置与数据一律参数注入）；
- `core/` **不得** import `adapters/`、`tune/`、`cli.py`。

> 这条直接根治 AM 的「双 config 架构倒置」缺陷（`engine.py` 反向 import 根目录 `config`）。

参考设计文档：`AlphaMaster-main/docs/miaosuan/ARCHITECTURE.md`（权威）与 `PRD.md`。

## 3. 快速开始

```bash
# 1. 创建虚拟环境并安装（含 dev 依赖与 pre-commit 钩子）
make env                  # 若 python 不在 PATH：make env PYTHON_BOOT=/path/to/python3.11

# 2. 最小自检（能 import 妙算包并跑通 vocab 自检）
make smoke

# 3. 全部测试 / 差分对拍
make test
make parity
```

Windows 上 Makefile 会自动使用 `.venv/Scripts/python.exe`；POSIX 使用 `.venv/bin/python`。

### 目录结构（MVP）

```
src/miaosuan/
├── __init__.py      # 包版本
├── config.py        # 单一配置树（dataclass + 依赖注入）
├── errors.py        # 统一异常体系 + 错误码
├── logging_setup.py # 结构化 JSON 日志（带 run_id）
└── core/            # ★纯 numpy 内核，无 IO / 无环境变量 / 无平台知识
    ├── registry.py  # 声明式注册层（FeatureSpec / OperatorSpec / Registry）
    ├── features.py  # 特征注册（M2 脚手架：仅名称；M4 填充 numpy 实现）
    ├── ops.py       # 算子注册（M2 脚手架：仅名称；M3 填充 numpy 实现）
    └── vocab.py     # ★VOCAB_VERSION 确定性派生 + verify（移植自 AM，近乎原样）
```

（`data/`、`market/`、`search/`、`gate/`、`ir/`、`adapters/`、`tune/`、`report/` 见后续任务 T02–T05。）

## 4. 词表版本恒等（移植正确性的零成本证明）

妙算的 `FormulaVocab.version` 沿用 AM 的确定性派生算法：

```
VOCAB_VERSION = "v" + sha256("\n".join(token_names)).hexdigest()[:12]
```

只要妙算的 token 组成与顺序和冻结的 AM 完全一致，版本字符串**必然相同**。
当前冻结值：

```
VOCAB_VERSION = v9217a2c0d91a       <- 与 AM 的 strategies/best_XAUUSD.json 一致
feature_count = 65
operator_count = 62
vocab_size     = 127
```

`tests/parity/test_vocab_identity.py` 在每次 CI 断言该恒等关系；若移植中不小心改了顺序或漏了算子，测试立刻变红。

## 5. 已知差异（Important）

| 项 | 架构目标 | 本机实际情况 | 处理 |
|---|---|---|---|
| **Python 版本** | **3.11**（架构 §8 锁定） | 本机仅有 3.13.12（`python --version` 报 3.13.14）与 3.14.3，**无 3.11**，且无 `install_binary` 工具可装 | 降级用 **3.13** 开发验证；`requires-python = ">=3.11"`，在 3.11 上应同样成立 |
| **numpy 版本** | `>=1.26,<2.1`（架构 §8） | numpy 1.26 无 cp313 wheel；3.13 需 `numpy>=2.1` | `requirements.lock` 按 **3.13 可安装**的实际版本锁定；在 3.11 上可回退到 `<2.1` |
| **算子数量** | 架构文档写 **66 个算子** | AM 实际 **62 个**（44 基础 + 3 跨截面 + 8 Task3.3 + 7 Task3.4） | **以 AM 实测为准 = 62**；`VOCAB_VERSION` 用 62 个算子派生才等于 `v9217a2c0d91a`。架构文档此处计数需勘误 |
| **tests/parity 的 Oracle** | 加载冻结的 AM（需 torch） | 本机未装 torch（AM `.venv` 亦缺） | 分两级：①**冻结快照**（`tests/fixtures/am_vocab_snapshot.json`，始终运行）；②**活体对拍**（装了 `oracle` extra 时自动启用，用真实 AM 模块比对）。详见 `tests/parity/conftest.py` |

## 6. 工程约定

- **密钥**一律走 `os.environ`，禁止硬编码；`pre-commit` 的 **gitleaks** 拦截（AM 曾有明文 token 泄露事故）。
- **异常**继承 `MiaoSuanError`，带稳定错误码（如 `E-VOCAB-MISMATCH`、`E-HOLDOUT-SEALED`）。
- **随机性**全部接受显式 `seed`，禁止隐式全局随机。
- **配置**单一来源：`src/miaosuan/config.py` 的 dataclass 树，依赖注入向下传递。

## 7. 与 AlphaMaster 的关系

- `AlphaMaster-main/` 仓库**只读冻结**，作为差分对拍的 **Oracle**（正确性基准），妙算**一行都不改它**。
- 妙算仓库与其**平级**，不嵌在其内部。
- 迁移边界与逐文件处置见架构文档 §1.4。

---

*当前进度：T01 的 M1（工程地基）+ M2（vocab 移植与对拍）。ops/features 的 numpy 化移植（M3/M4）为后续任务。*
