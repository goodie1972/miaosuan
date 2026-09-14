# 妙算（MiaoSuan）

> **一套纯 numpy 的声明式因子语言 + 一个遗传编程搜索器 + 一个 神机 策略双向适配层。**

---

## 1. 妙算是什么

妙算是一套**离线**的量化研究与策略生产工具，聚焦两件事：

| 模式 | 输入 | 输出 | 引擎 |
|---|---|---|---|
| **模式 A：因子挖掘** | 行情数据（Panel） | 一份可信的因子公式 → 可导出的 神机 策略 | RPN-GA 结构搜索 |
| **模式 B：参数寻优** | 已有 神机 策略 | 外提参数后的最优参数集 → 回写策略新版本 | Optuna TPE + 事件驱动回测 |

两种模式通过**统一的产物 IR（`StrategySpec`）** 打通：模式 A 的导出物可直接作为模式 B 的输入，
形成 `mine → export → tune → export v2 → tune …` 的闭环。

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
4. **历史包袱消除**：原项目中唯一依赖 torch 的模块（基于 RL 的公式搜索器）已被遗传编程搜索器完全替代，torch 不再出现在生产依赖中。

> `requirements.lock` 锁定全部依赖版本，`pyproject.toml` 声明 `requires-python = ">=3.11"`，确保跨环境一致。

## 3. 架构一览（Hexagonal-lite）

```
CLI  ──►  Pipeline  ──►  {search(GA) | tune(Optuna)}  ──►  core(纯 numpy)
                          │                                  ▲
                    data/ + market/（可插拔）             ir/ + gate/ + report/
                          │
                    adapters/shenji（正向生成 / 反向抽取）
```

**依赖方向铁律（CI 强制）**：

- `core/` **不得** `import torch`；
- `core/` **不得** 读环境变量、**不得** 做文件 IO（配置与数据一律参数注入）；
- `core/` **不得** import `adapters/`、`tune/`、`cli.py`。

## 4. 快速开始

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

## 5. 词表版本恒等

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
| **Python 版本** | **3.11** | 本机仅有 3.13.12 与 3.14.3，无 3.11 | 降级用 **3.13** 开发验证；`requires-python = ">=3.11"` |
| **numpy 版本** | `>=1.26,<2.1` | numpy 1.26 无 cp313 wheel；3.13 需 `numpy>=2.1` | `requirements.lock` 按 **3.13 可安装**的实际版本锁定 |
| **算子数量** | 66 个 | 实际 **62 个**（44 基础 + 3 跨截面 + 8 Task3.3 + 7 Task3.4） | 以实测为准 = 62；`VOCAB_VERSION` 用 62 个算子派生 |

---

## License

Proprietary — All rights reserved.

---

*当前进度：T01–T07 完成（工程地基 + vocab 移植 + ops/features numpy 化 + 端到端对拍 + 神机适配器 + 回测 + Web UI）。*
