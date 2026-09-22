# 妙算（MiaoSuan）

> **一套纯 numpy 的声明式因子语言 + 一个遗传编程搜索器 + 一个 神机 策略双向适配层。**

---

## 1. 妙算是什么

## 核心特色概览

- 插件化架构：适配器层、搜索器插件、调优插件，实现可插拔、零耦合。
- 可扩展的市场画像：冻结的 `FrozenMarketProfile` 零代码新增多市场。
- 纯 NumPy 实现：轻量、跨平台、保证结果可复现，彻底摆脱 Torch 依赖。
- 统一的 IR 与 CI 检查：`StrategySpec`、严格的测试与对拍保障。

妙算是一套**离线**的量化研究与策略生产工具，聚焦两件事：

| 模式 | 输入 | 输出 | 引擎 |
|---|---|---|---|
| **模式 A：因子挖掘** | 行情数据（Panel） | 一份可信的因子公式 → 可导出的 神机 策略 | RPN-GA 结构搜索 |
| **模式 B：参数寻优** | 已有 神机 策略 | 外提参数后的最优参数集 → 回写策略新版本 | Optuna TPE + 事件驱动回测 |

两种模式通过**统一的产物 IR（`StrategySpec`）** 打通：模式 A 的导出物可直接作为模式 B 的输入，
形成 `mine → export → tune → export v2 → tune …` 的闭环。

> **模式 B 额外依赖**：参数寻优走 Optuna，位于 `tune` extra 且**默认不安装**。
> 使用 `miaosuan tune` 前先执行 `pip install -e ".[tune]"`。

## 2. 为什么用纯 Numpy（不用 Torch）？

| 维度 | 纯 numpy 方案 | torch 方案 |
|---|---|---|
| **安装体积** | ~100 MB（numpy + pandas + scipy） | ~600 MB（torch CPU）~ 2 GB（CUDA 版） |
| **环境可复现** | ✅ Python 3.11 + locked requirements，跨平台一致 | ❌ GPU/CPU 版本冲突、CUDA 驱动绑定 |
| **部署门槛** | ✅ 任意机器 `pip install` 即可用 | ❌ 需匹配 CUDA toolkit / 驱动版本 |
| **核心算法适配** | ✅ 全部是 elementwise、滑动窗口、归约——numpy 原生高效 | 杀鸡用牛刀：本系统无梯度反传需求 |

**技术决策说明**：

1. **无梯度需求**：妙算的因子搜索采用遗传编程（RPN-GA），不需要梯度反传；特征与算子计算全部是前向的 elementwise / sliding-window / reduction 操作，numpy 原生即可高效完成。
2. **环境复现**：量化研究对可复现性要求极高——同一份代码在不同机器上必须产出逐位一致的结果。torch 的 GPU/CPU 变体、CUDA 版本矩阵让这一点几乎不可能保证；numpy 的 float32 运算在主流 CPU 上行为确定。
3. **轻量部署**：妙算面向策略工厂场景，需要在 CI、笔记本、服务器上快速部署；100 MB 的安装体积远优于 600 MB+ 的 torch 栈。
4. **从一开始就使用 NumPy**：我们了解 Torch 的缺陷，选择了更轻量、可复现的 NumPy 方案，所有核心实现均基于纯 NumPy，未再引入 Torch 依赖。

> `requirements.lock` 锁定全部依赖版本，`pyproject.toml` 声明 `requires-python = ">=3.11"`，确保跨环境一致。

## 3. 架构一览（Hexagonal-lite）

```
CLI  ──►  Pipeline  ──►  {search(GA) | tune(Optuna)}  ──►  core(纯 numpy)
                          │                                  ▲
                  data/ + market/（可插拔）             ir/ + gate/ + report/
                        │
                  adapters/shenji（正向生成 / 反向抽取）
                  data/ 下的数据获取模块统一封装「神机本地库 + 网络下载」两类来源（§3.2）
```

**依赖方向铁律（CI 强制）**：

- `core/` **不得** `import torch`；
- `core/` **不得** 读环境变量、**不得** 做文件 IO（配置与数据一律参数注入）；
- `core/` **不得** import `adapters/`、`tune/`、`cli.py`。

### 3.1 神机-妙算配套关系与部署边界

妙算（MiaoSuan）与神机（Shenji）是**配套套装**，二者分工明确、通过统一接口解耦：

- **神机**：策略运行与执行侧。**正式上线部署运行在神机侧**——实盘 / 半实盘交易、信号分发、实时行情接入均在神机后台完成。
- **妙算**：策略研究与生产侧。负责因子挖掘、参数寻优、回测、纸面交易仿真，产出可导出的神机策略。

妙算的产物（神机策略 `.py`）经 `export` 导出后交由神机加载运行；神机运行产生的绩效、信号、实时行情又回流为妙算回测与纸面交易的输入。两者通过统一的 `StrategySpec` IR 与数据获取模块解耦，**互不内嵌**。

### 3.2 数据获取模块（统一两类来源）

妙算所需历史数据来源于**两类**，必须统一封装为一个**独立的数据获取模块**（建议落地于 `src/miaosuan/data/acquisition.py` 或独立 `datafeed/` 包），对上层（`loader`、CLI、纸面交易）只暴露统一的 `fetch(symbol, timeframe, since=None)` 接口：

| 来源 | 说明 | 封装方式 |
|------|------|----------|
| **① 神机本地数据库** | 神机侧已落库的行情（首选、最低延迟） | 经神机提供的只读查询接口 / 本地 DB 驱动拉取 |
| **② 网络下载** | 从数据服务商 / 公开源按需下载 | 经封装好的 HTTP/SDK 客户端拉取 |

**职责边界**：

- 数据获取模块**只负责"取数"**：统一抽象两类来源、处理认证、缓存落地、增量 / 全量决策。**不**做特征计算、不碰策略逻辑、不写回神机。
- 取回的原始数据统一落盘为 parquet（沿用 `loader` 的 `infer_symbol_timeframe` 命名约定），后续由 `loader.load()` 加载为 `Panel`。
- 上层（挖掘 / 回测 / 纸面交易）**不直接接触**来源细节，只调用 `fetch(...)`；来源切换（如本地库暂不可用改走网络）对上层透明。

**数据流向**：

```
[神机本地库] ─┐
             ├─► [数据获取模块 fetch()] ─► 本地缓存(parquet) ─► loader.load() ─► Panel ─► 挖掘/回测/纸面
[网络下载]  ─┘        (增量优先 / 首跑全量)
                                          ▲
                              [纸面交易]───┘ (实时行情经神机后台回流)
```

**增量优先获取策略**：

- **优先增量**：本地已存在该品种原始数据缓存时，网络下载只拉取「本地最新时间戳 → 现在」的增量部分并追加，避免重复传输全量历史。
- **首跑退化**：若本地**尚不存在**原始数据缓存（首次接入该品种），则本次增量获取**自动退化为全量获取**，拉取完整历史后再转入常态增量模式。
- 增量 / 全量的判定由模块内部基于本地缓存的存在性与 `max(time)` 自动完成，调用方无需感知。

**部署说明**：

- 数据获取模块**随妙算部署**，但正式生产环境的"数据源主干"指向**神机本地数据库**；网络下载作为补充 / 灾备通道。
- 神机侧需为妙算开放**只读**数据查询权限（账号 / 令牌走 `os.environ`，禁止硬编码，受 gitleaks 拦截）。
- 网络下载通道建议配置合理限流与重试；增量模式注意服务商时间窗限制。

### 3.3 纸面交易对接神机后台

妙算内的**纸面交易**同样**对接神机后台**：以神机后台提供的实时 / 仿真行情与执行回执为输入，在妙算侧做信号复现与绩效仿真，**不**独立搭建行情源。

- 纸面交易的行情源、成交回执、账户状态均经神机后台只读接口回流，与正式上线共享同一数据通路。
- 这保证了「纸面」与「实盘」在数据口径、成本模型、时间戳语义上完全一致，纸面验证通过的策略可直接移交神机上线，避免环境偏差。

## 4. 项目特色

### 插件化架构

- **适配器层 (`adapters/shenji`)**：实现对外部数据源、交易所 SDK、模型平台的可插拔接入，保持 `core/` 完全独立。
- **搜索器插件**：遗传编程搜索器位于 `search/ga.py`，通过统一的 `SearchEngine` 接口，可随时替换为其他搜索策略（如强化学习、贝叶斯优化），无需修改核心代码。
- **调优插件**：`tune/` 目录提供基于 Optuna 的参数寻优插件，同样遵循统一接口，实现“一键”切换调优算法。

### 可扩展的市场画像

- 市场画像采用 **冻结的 `FrozenMarketProfile`** 实例化方式，实现 **零代码** 新增。
- 只需在 `src/miaosuan/market_profiles/` 添加一个 Python 文件，声明 `profile = FrozenMarketProfile(...)` 即可在 CLI/代码中使用。
- 支持自定义 **费用模型、杠杆、可交易掩码、时间戳单位、成交量语义** 等，满足外汇、股票、加密货币等多品类需求。

### 轻量化、可复现、无 Torch 依赖

- 完全基于 **pure NumPy** 实现，避免 GPU/CPU 版本冲突，确保跨平台结果一致。
- `requirements.lock` 锁定依赖版本，`pyproject.toml` 声明 `requires-python = ">=3.11"`，在任意环境 `pip install -r requirements.lock` 即可部署。

  > 注：本机锁文件名为 **`requirements.lock`**（不存在 `requirements.txt`）。完整开发安装（含 Web UI 与可选数据源）：
  > `pip install -e ".[dev,web,datasource]"`。

---


```bash
# 1. 创建虚拟环境并安装（含 dev 依赖与 pre-commit 钩子）
make env                  # 若 python 不在 PATH：make env PYTHON_BOOT=/path/to/python3.11

# 2. 最小自检（能 import 妙算包并跑通 vocab 自检）
make smoke

# 3. 全部测试 / 回归对拍
make test
make parity
```

Windows 上 Makefile 会自动使用 `.venv/Scripts/python.exe`；POSIX 使用 `.venv/bin/python`。

### 目录结构

```
src/miaosuan/
├── __init__.py      # 包版本
├── config.py        # 单一配置树（dataclass + 依赖注入）
├── errors.py        # 统一异常体系 + 错误码
├── logging_setup.py # 结构化 JSON 日志（带 run_id）
├── cli.py           # CLI 入口（mine → export → verify → report → backtest → ui）
└── core/            # ★纯 numpy 内核，无 IO / 无环境变量 / 无平台知识
    ├── registry.py  # 声明式注册层（FeatureSpec / OperatorSpec / Registry）
    ├── features.py  # 特征注册（65 维 numpy 实现）
    ├── ops.py       # 算子注册（62 个算子 numpy 实现）
    ├── vocab.py     # ★VOCAB_VERSION 确定性派生 + verify
    ├── vm.py        # StackVM 公式解释执行
    ├── evaluator.py # 有效性评估（夏普 / 衰减比 / 样本外门禁）
    └── backtest.py  # 回测引擎（tanh 仓位 + 成本模型）
```

## 5. 快速开始

```bash
# 1. 创建虚拟环境并安装（含 dev 依赖与 pre‑commit 钩子）
make env                  # 若 python 不在 PATH：make env PYTHON_BOOT=/path/to/python3.11

# 2. 最小自检（能 import 妙算包并跑通 vocab 自检）
make smoke

# 3. 全部测试 / 回归对拍
make test
make parity
```

Windows 上 Makefile 会自动使用 `.venv/Scripts/python.exe`；POSIX 使用 `.venv/bin/python`。

妙算的 `FormulaVocab.version` 采用确定性派生算法：

```
VOCAB_VERSION = "v" + sha256("\n".join(token_names)).hexdigest()[:12]
```

只要 token 组成与顺序不变，版本字符串**必然相同**。当前冻结值：

```
VOCAB_VERSION = v9217a2c0d91a
feature_count = 65
operator_count = 62
vocab_size     = 127
```

`tests/parity/test_vocab_identity.py` 在每次 CI 断言该恒等关系；若不小心改了顺序或漏了算子，测试立刻变红。

## 6. CLI 工作流

```bash
# 因子挖掘 → 导出策略 → 验证 → 报告 → 回测 → Web UI
miaosuan mine --data XAUUSD_H1.parquet --profile FOREX_XAUUSD
miaosuan export --strategy best_XAUUSD.json --profile FOREX_XAUUSD
miaosuan verify --strategy exported_strategy.py
miaosuan report --strategy exported_strategy.py --format html
miaosuan backtest --strategy exported_strategy.py --data XAUUSD_H1.parquet
miaosuan ui --port 8686
```

## 7. 市场画像（Market Profile）

妙算采用**零接触可扩展**的交易规则容器 `FrozenMarketProfile`，每个市场实例封装：

- 成本模型（手续费、滑点、最小点值）
- 杠杆与保证金规则
- 可交易时段掩码
- 年化期数（`periods_per_year`）
- 成交量语义（tick / real）

已内置画像：`FOREX_XAUUSD`、`CN_EQUITY_RESEARCH`、`US_EQUITY_RESEARCH`、`CRYPTO_BTC`。

新增市场只需声明一个 `FrozenMarketProfile` 实例，**无需修改 core 任何代码**。

## 8. 工程约定

- **密钥**一律走 `os.environ`，禁止硬编码；`pre-commit` 的 **gitleaks** 拦截。
- **异常**继承 `MiaoSuanError`，带稳定错误码（如 `E-VOCAB-MISMATCH`、`E-HOLDOUT-SEALED`）。
- **随机性**全部接受显式 `seed`，禁止隐式全局随机。
- **配置**单一来源：`src/miaosuan/config.py` 的 dataclass 树，依赖注入向下传递。

## 9. 已知差异

| 项 | 架构目标 | 本机实际情况 | 处理 |
|---|---|---|---|
| **Python 版本** | **3.11** | 本机仅有 3.13.14 与 3.14.3，无 3.11 | 降级用 **3.13** 开发验证；`requires-python = ">=3.11"` |
| **numpy 版本** | `>=1.26,<2.1` | numpy 1.26 无 cp313 wheel；3.13 需 `numpy>=2.1` | `requirements.lock` 按 **3.13 可安装**的实际版本锁定 |
| **算子数量** | 66 个 | 实际 **62 个**（44 基础 + 3 跨截面 + 8 Task3.3 + 7 Task3.4） | 以实测为准 = 62；`VOCAB_VERSION` 用 62 个算子派生 |

---

## License

GNU Affero General Public License v3 (AGPL-3.0) — 完整许可证文本见 LICENSE 文件。

---

*当前进度：T01–T07 完成（工程地基 + vocab 移植 + ops/features numpy 化 + 端到端对拍 + 神机适配器 + 回测 + Web UI）。*
