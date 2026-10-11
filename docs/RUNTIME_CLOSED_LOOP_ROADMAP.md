# 妙算（MiaoSuan）2.0 运行时闭环升级路线 —— MT5 接入 / 实时信号 / 纸面与实盘执行

| 项目 | 内容 |
|---|---|
| 文档类型 | 升级设计（可执行路线图 + 样例代码） |
| 作者 | 架构师 |
| 版本 | v1.1（评审修订版） |
| 日期 | 2026-10-06（v1.0：2026-10-04） |
| 上游输入 | 2026-10-04 用户需求「2.0 达成研究→执行闭环：对接 MT5、产出实时信号」+ 架构评审结论 + 三轮评审意见 |
| 关联代码 | `src/miaosuan/ir/schema.py`、`src/miaosuan/adapters/`、`src/miaosuan/data/`、`src/miaosuan/gate/`、`src/miaosuan/webui/`、`src/miaosuan/search/`、`pyproject.toml` |
| 性质 | **设计文档；samples 为新增文件骨架或既有文件的增量改动示意（如 Sample A 仅追加字段与两行序列化），不重写既有逻辑** |
| v1.1 修订 | R 系列：Sample A 改为 schema.py 三处同步 + spec_id 漂移策略；Sample C/E 改用 `cost_model`/`BrokerSpec` 并修 min_lot 向上取整风险；ExecutionPort 补平仓与成交回执；G1 命名契约对齐 `acquisition.py`；依赖白名单补 `core.vocab`/`gate`。S 系列：Sample I 落盘模式修复；"精英只在单代内"表述纠正；冻结基线重新基线化入工作量；`rpn.py` 补入模块表。新增附录 A（M5 功能对照） |
| 豁免记录 | 2026-10-06 R1 合并前核查：`core/` 6 个文件存在 2026-10-05 11:31 的**注释级**词替换（"妙算"→"对拍基准"，共 14 行、无任何行为变更，同批波及 `search/`、`pipeline.py`、`tests/parity/`），owner 裁定**豁免**（即日生效）；G6 口径据此修订为「`core/` 无行为变更，注释级改动登记豁免」。此后的任何 `core/` diff 仍须逐一审查 |

> 核心判断：**妙算离闭环的差距不在研究侧，而在"策略产出之后没有运行时"。**
> 数据源抽象、信号语义 IR（`Semantics`）、部署门禁、对拍文化都已就位；
> 缺的是一个与 `cli`/`webui` 同级、允许 IO 的 **runtime 层**，把 StrategySpec 变成
> 能在收盘 bar 上持续出信号、能下单、能对账、能熔断的活系统。
> 全程不动 `core/` 一行代码，是本路线的硬约束。

---

## 0. 结论摘要（先说答案）

| # | 问题 | 路线结论 |
|---|---|---|
| 1 | 闭环缺什么？ | 缺 **runtime 层**：收盘调度、信号引擎、风险仓位、执行路由、状态持久化、对账熔断、通知观测。算法与回测侧零缺口。 |
| 2 | 从哪先动手？ | **R1 先做 MT5 数据源**（不碰实盘即可让挖掘/回测受益），再 R2 信号引擎+对拍，R3 纸面，R4 实盘，R5 观测。顺序不可换。 |
| 3 | 最大风险？ | **口径漂移**：实时侧与回测侧的信号不一致（未收盘 bar、执行延迟、中性带不一致）。对策：把口径全部写进 `Semantics` IR + 对拍上 CI，而不是靠自觉。 |
| 4 | MT5 依赖怎么处理？ | `MetaTrader5` 进 `[live]` 可选 extra（Windows only），与 `tune` extra 同一模式；runtime 在缺依赖时优雅降级为信号/纸面模式。 |
| 5 | 执行侧架构？ | **三个可插拔 ExecutionPort 实现**：PaperBroker（默认）、MT5Broker、ShenjiRouter（转发神机后台）。运行时不知道单下到了哪，"配套神机"的定位不变。 |
| 6 | 门禁在闭环里的位置？ | **前置硬门**：runtime 启动策略时校验 `gate_verdict == DEPLOYABLE` 且 `vocab_version` 匹配，否则拒绝启动（`E-RUNTIME-NOT-DEPLOYABLE`）。 |
| 7 | 工作量？ | R 系列全程约 **3~4 周**（R1 2~3 天，R2 3~4 天，R3 4~5 天，R4 5~6 天**+模拟户手工验收缓冲 2 天**，R5 2~3 天），R3/R4 的生产化细节（对账/恢复/熔断）占一半以上。 |
| 8 | 搜索侧还有缺口吗？ | 有——**长任务续训能力空白**：mine 是一次性批处理，中断即全废，跨次运行无分数下限。补 **S 系列**（S1~S6，约 1.5~2 周，与 R 系列并行）：检查点续挖、分数下限、精英龄期回放、因子池相关性惩罚、代内成本压力、token 剪枝。原版 RL miner 17 项机制的逐项对照见附录 A。 |

---

## 1. 背景与问题定义

妙算当前的产品边界是「策略研究 → 策略产物（StrategySpec / 导出 .py）」。
生产部署依赖神机侧执行，这个定位不变；但存在三个现实问题：

1. **验证断层**：纸面交易依赖神机后台回流数据，没有神机环境时，策略从"回测通过"到"实盘上线"之间没有任何本机验证手段。
2. **数据断层**：挖掘与回测依赖既有 parquet 缓存，接入新品种/新周期要靠手工取数；`data/fetchers/` 里 8 个 fetcher 中唯独缺 MT5 直连（已有 `mt4_bridge.py` 走 EA 桥的先例）。
3. **观测断层**：webui 实时页是神机后台的只读客户端；本地策略没有自己的实时信号视图与通知通道。

结论：**需要一个 runtime 层补上"策略的装卸码头"**，它本身不做研究、不算因子，
只做调度、执行、风控、状态与观测。

## 2. 目标与非目标

### 2.1 目标（可度量）

| 编号 | 目标 | 度量方式 |
|---|---|---|
| **G1** | MT5 数据源可用：`fetch(symbol, timeframe)` 支持全量+增量自动切换，落盘 parquet 符合 acquisition **既有命名契约** `{symbol}_{timeframe}_{source_type}_{note}.parquet`（显式指定 source 时 source_type 进文件名，如 `XAUUSD_M30_MT5.parquet`，见 `acquisition.py`） | 集成测试：断网场景走本地缓存、联网场景增量拉取，两种路径各有断言 |
| **G2** | 任一 StrategySpec 在收盘 bar 上实时出信号，且与回测引擎**逐 bar 一致** | 对拍测试：同一份历史 bar，回测仓位序列 == 实时信号仓位序列（逐元素相等） |
| **G3** | 纸面模式连续运行 ≥ 2 周：账本与成交回执对账差异为 0 | 运行报告 + 对账断言（`E-BROKER-RECONCILE` 不触发） |
| **G4** | MT5 实单链路可用：下单、平仓、持仓同步、对账、紧急熔断、kill 开关；**未过 `DEPLOYABLE` 门禁的策略拒绝启动** | 手工验收清单 + 门禁单元测试 |
| **G5** | 观测闭环：webui 实时页支持本地 runtime 后端；飞书 Webhook 在信号方向转折时推送 | e2e 冒烟脚本 |
| **G6** | 架构纯度：`core/` **无行为变更**（diff 为空；注释级改动须在此登记豁免） | CI 检查：`grep -r "MetaTrader5" src/miaosuan/core/` 为空 + diff 审查 |
| **G7** | S 系列长任务鲁棒性：检查点续挖无损、分数下限防覆盖、搜索动力学变更后基线受控 | 故障注入测试 + 下限保护单测 + 基线重跑清单 |

### 2.2 非目标（明确不做）

- **不做外部策略静默导入**：词表版本不匹配的策略一律显式拒绝（沿用 `E-VOCAB-MISMATCH`），不做自动映射/翻译。跨词表兼容层留待有真实需求时单独立项。
- **不做多账户/组合级资金管理**：本路线只做单品种仓位缩放（RiskPolicy），组合层留给后续版本。
- **不修改 core/ 与 IR 四段结构**：`Semantics` 只允许**追加字段**（见 Sample A 的兼容策略与 spec_id 处理）。
- **不自建行情源主干**：延续 `acquisition.DataSource` 抽象，神机本地库仍是生产首选，网络下载为灾备，MT5 为第三通道。
- **不复刻原版 RL 训练动力学**（熵/KL 损失、LoRD/StableRank、策略网络梯度回放）：GA 架构下以多样性硬约束与停滞移民承担等价的探索压力，逐项对照见附录 A。

## 3. 总体架构

```
CLI / webui ──► pipeline ──► {mine(GA) | tune(Optuna)} ──► core(纯 numpy)   ← 一行不改
                                  │
                                  ▼ StrategySpec(.json)
                    ┌─────────────────────────────────────────────┐
                    │           runtime/（新增，允许 IO）           │
                    │                                             │
                    │  scheduler     周期收盘调度（只吃已收盘bar）  │
                    │  signal_engine StackVM + Semantics 出信号    │
                    │  risk          RiskPolicy（波动率目标仓位）   │
                    │  state         账本持久化/崩溃恢复/kill开关   │
                    │  observer      指标/日志/通知分发            │
                    │                                             │
                    │  ports（新增端口，与 adapters.TargetPort 并列）│
                    │   ├─ DataPort      ─► mt5 / mt4桥 / shenji   │
                    │   ├─ ExecutionPort ─► paper / mt5 / shenji   │
                    │   └─ NotifierPort  ─→ feishu / log           │
                    └─────────────────────────────────────────────┘
```

**依赖方向铁律（新增两条，CI 强制）：**

- `runtime/` 不得 import `core/` 之外的任何研究模块内部实现；消费策略只经
  `ir.schema.StrategySpec` + `core.vm` / `core.vocab`（`FORMULA_VOCAB.verify`）公开接口，
  门禁校验只读 `gate` 的 verdict 结论（见 Sample G）——白名单即此四项，逐字进 CI 检查；
- `core/` 不得 import `runtime/`、`adapters/`；`adapters/` 不得 import `runtime/`。

**与神机的配套关系保持不变**：`ShenjiRouter` 只是 ExecutionPort 的一个实现，
正式生产仍可将执行路由到神机后台；MT5 直连是本机验证与轻量实盘的补充通道。

## 4. 关键设计决策

| # | 决策 | 理由 |
|---|---|---|
| D1 | runtime 与 cli/webui 同级，**不进 core** | 实盘代码腐烂速度远快于研究代码；隔离才能保住 core 的纯度与对拍文化 |
| D2 | 口径全部写进 `Semantics`（新增 `execution_lag_bars` 等字段） | 口径一致性是产品属性，必须被 IR 版本守护 + 对拍测试证明，不能散落在配置注释里；`execution_lag_bars=1` 与回测执行口径 `target_ret[t] = log(open[t+2]/open[t+1])`（bar t 收盘出信号 → t+1 开盘执行）一致 |
| D3 | 门禁前置为启动硬校验 | 带 `RESEARCH_ONLY`/`BLOCKED` verdict 的策略进 runtime 直接抛 `E-RUNTIME-NOT-DEPLOYABLE` |
| D4 | `MetaTrader5` 进 `[live]` extra，Windows only | 与 `tune` extra 同模式；Linux 服务器/CI 不装也能跑信号与纸面模式 |
| D5 | 执行三实现可插拔，运行时无感知 | 保住"神机配套"的既有定位；MT5 直连只是多一个 adapter |
| D6 | 状态持久化 + 对账 + kill 文件是 R3/R4 的**验收核心** | 顺序：能恢复 → 能对上账 → 能急停。三者缺一不得开实单 |
| D7 | 通知走 NotifierPort，飞书为第一个实现 | 转折推送是高频使用的观测手段，但必须可替换（邮件/webhook 后续零成本接入） |
| D8 | 成本与经纪商规格的单一事实：成本走 `cost_model`（CostModel），规格走 `BrokerSpec` | 画像层不另设费用字段，杜绝"两个成本事实来源"；`BrokerSpec` 是画像的**可选字段**，既有画像零改动 |

## 5. 分阶段步骤计划

> 每阶段交付物均可独立合并主干；阶段内任务按依赖排序；验收标准是合并门槛。

### R1 — MT5 数据源 + 市场画像（2~3 天）

| 任务 | 内容 |
|---|---|
| R1.1 | 新增 `data/fetchers/mt5_src.py`：实现 `DataSource`（`fetch_full` / `fetch_incremental` / `is_available`），复用 `acquisition` 的校验、缓存、增量判定逻辑 |
| R1.2 | 处理**经纪商时间戳**：沿用 `mt4_bridge.py` 的 UTC 偏移探测方案（`detect_utc_offset_hours` 思路），输出口径统一 UTC epoch 秒 |
| R1.3 | 市场画像扩展（落点 `market/profiles.py`，非独立目录）：新增 `FOREX_XAUUSD_MT5` 画像（命名对齐既有 `FOREX_XAUUSD` 惯例）；**成本沿用既有 `cost_model`（CostModel）单一事实**；经纪商品种规格（tick_value、lot_step、min_lot、contract_size）新增 `BrokerSpec` frozen dataclass，作为画像**可选字段** `broker_spec`（默认 None，既有画像零改动）。⚠ `MARKET_PROFILE_REQUIRED_FIELDS = MARKET_PROFILE_FIELDS[:-1]` 依赖"仅末位一个可选字段"的切片假设，追加字段时必须同步改为显式可选集合，否则 `limit_pct` 会被误升为必填 |
| R1.4 | `pyproject.toml` 增加 `[live]` extra：`MetaTrader5>=5.0.45 ; sys_platform=='win32'` |
| R1.5 | 集成测试：联网增量 / 断网走缓存 / 命名契约三条断言（契约按 G1 的 `{symbol}_{timeframe}_{source_type}…` 口径断言） |

**验收**：`miaosuan` 侧一行命令完成 `fetch("XAUUSD","M30")` 全量+增量；挖掘与回测可直接吃新数据（R1 单独即可让研究侧受益）。

### R2 — runtime 骨架 + 信号引擎 + 对拍（3~4 天）

| 任务 | 内容 |
|---|---|
| R2.1 | 新增 `src/miaosuan/runtime/` 包：`scheduler.py`（周期收盘检测、去重、断线补拉）、`signal_engine.py`（加载 StrategySpec → 校验门禁/词表 → 收盘 bars → StackVM → 仓位序列）、`ports.py`（DataPort/ExecutionPort/NotifierPort 抽象，见 Sample B） |
| R2.2 | `ir/schema.py` 的 `Semantics` 追加 `execution_lag_bars`（默认 1，D2 口径）；实时侧观望阈值**直接复用 `semantics.neutral_band`**（0.05，与 `MIN_TRADE_EXPOSURE` 同值且已由 cli.py 映射接线），**不另立 min_exposure 配置**。改动点是 `schema.py` 的 `to_dict`/`from_dict` 各加一行（不是 codec.py），外加 spec_id 哈希策略——三处缺一不可，见 Sample A |
| R2.3 | watchlist：`miaosuan watch add XAUUSD --spec best.json --profile FOREX_XAUUSD_MT5` |
| R2.4 | **对拍 harness**（本路线的验收基石，见 Sample F）：录制/回放 FakeBroker，同一份 bar 喂回测引擎与实时信号路径，断言逐 bar 仓位相等；进 CI |
| R2.5 | 信号模式冒烟：本机跑通"收盘 → 信号 → 日志/通知"，不下单 |

**验收**：G2 的对拍断言在 CI 常绿；watchlist 三个品种连续 24h 无漏拍、无重复拍。

### R3 — 纸面模式 + 风险仓位 + 状态持久化（4~5 天）

| 任务 | 内容 |
|---|---|
| R3.1 | `risk.py`：`RiskPolicy` 协议 + `VolTargetPolicy` 实现（波动率目标仓位，纯 numpy，见 Sample E）；Sharpe 权重、ATR 基准来自 RiskPolicy 配置，lot_step/min_lot/contract_size 取自 `BrokerSpec`；**目标手数低于 min_lot 时不交易（返回 0），禁止向上取整到 min_lot** |
| R3.2 | `brokers/paper.py`：`PaperBroker`（**事件式撮合**：submit 入队 → 下一根已收盘 bar 开盘价 + `cost_model` 成本成交，见 Sample C） |
| R3.3 | `state.py`：账本 JSON 落盘（原子写 + 单实例锁 `E-RUNTIME-STATE-LOCK`）、崩溃恢复（启动时重放未完成意图）、kill 文件检测（`STOP` 文件存在即平仓停机） |
| R3.4 | `cli.py` 接线：`miaosuan paper start/stop/status` |
| R3.5 | 对账器 `reconcile.py`：本地账本 vs 撮合回执（Paper=ledger；MT5=`history_deals_get`，经 `ExecutionPort.fills`）逐笔比对，差异抛 `E-BROKER-RECONCILE` |

**验收**：G3（两周纸面 + 零对账差异）；kill 文件 ≤ 1 个调度周期内生效（测试断言）。

### R4 — MT5 实盘执行（5~6 天 + 验收缓冲 2 天）

| 任务 | 内容 |
|---|---|
| R4.1 | `brokers/mt5.py`：`MT5Broker` 实现 ExecutionPort（懒加载 MetaTrader5，见 Sample D）：下单、**按 ticket 平仓**（hedging 账户反向 submit 是新开仓不是平仓）、持仓查询、成交回执（`history_deals_get`）、品种规格读取；市价单带 `price`/`deviation`/`type_time` |
| R4.2 | 对账增强：MT5 持仓与本地账本的方向/手数/魔法号一致性校验（走 `ExecutionPort.fills` 回执），不一致先平后报 |
| R4.3 | 熔断：单日最大亏损、极端滑点、连续拒单 N 次 → 自动平仓 + `E-BROKER-KILLSWITCH` + 飞书告警 |
| R4.4 | 启动安全链：门禁校验（D3）→ kill 文件检查 → 账本恢复 → 才允许挂单；`miaosuan live start` 需显式 `--i-understand-real-money` |
| R4.5 | 手工验收清单：模拟户全流程（开/平/部分成交/拒单/断线重连/kill） |

**验收**：G4；R4 合并前 R3 的对账必须在真实行情下连续一周零差异。

### R5 — 观测与通知（2~3 天）

| 任务 | 内容 |
|---|---|
| R5.1 | webui 实时页扩展：数据后端从"只读神机"扩为 `local | shenji` 双通道；本地通道展示 watchlist 信号卡（方向/把握/最近调仓） |
| R5.2 | `notifiers/feishu.py`：方向转折推送（脱敏：只发文字信号，不发持仓金额）；`notifiers/log.py` 兜底 |
| R5.3 | e2e 冒烟脚本进 `scripts/`：起服务 → watch → 收盘信号 → 通知 mock → 断言 |

**验收**：G5；页面与通知在 R2 的信号模式上即可先行联调。

## 6. S 系列 — 搜索侧续训与反停滞升级（与 R 系列并行，约 1.5~2 周，含 1 天基线重冻结）

> R 系列解决"策略产出之后"，S 系列解决"策略产出之前的长任务鲁棒性"。
> 现状：`mine` 是一次性批处理——进程中断即全废；跨次运行没有分数下限保护，
> 弱结果可以覆盖既有最优。GA 的精英**虽已跨代保留**（deepcopy 进下一代岛），
> 但没有龄期老化与回放——老化精英长期占据池位，`Individual.birth_gen` 字段
> 已存在却未被利用。全部机制纯 numpy，与 torch 无关，不触碰依赖铁律。

| 编号 | 机制 | 任务 | 验收标准 |
|---|---|---|---|
| **S1** | **搜索检查点 + 断点续挖** | 每隔 K 代把种群、精英池、预算状态、RNG 状态原子落盘（`artifacts/mine_ckpt/`，骨架见 Sample I）；`mine` 启动时检测到检查点自动续挖，代数计数与最优分无损；**恢复后先执行一次 S3 精英回放，再继续演化** | 故障注入测试：第 N 代 kill -9（Windows 用 `taskkill /F` 等价）后恢复，`n_evaluations` 单调不减、最优分不低于中断前 |
| **S2** | **分数下限** | 读取 artifacts 中同标的既有最优的 `evidence.val_score` 作为下限注入搜索器；仅**严格更高分**才更新产物文件，弱结果一律不覆盖（日志明示"未覆盖"） | 单测：下限=0.74 时，0.73 的候选不落盘、0.75 落盘 |
| **S3** | **精英龄期回放** | 启用既有 `Individual.birth_gen`（ga.py 已有字段）；每代从池中按「归一化分数软温度 softmax（T<1 防单点垄断）× 半衰衰减权重」采样 n_replay 条注入当代评估与选择，并**按 tokens 去重**，防止池化老化 | 单测：衰减权重随代龄指数下降；采样分布不退化为 argmax |
| **S4** | **因子池相关性惩罚** | 维护 top-k 因子向量池；新公式与池内任一因子 \|corr\| 超阈值则在适应度上乘罚系数。**相关性只在 train 折切片上计算**，杜绝 val 信息泄漏 | 泄漏单测：人为污染 val 段不影响惩罚判定；阈值边界测试 |
| **S5** | **代内成本压力测试 + 稀疏防刷分** | 适应度中加入「2× 成本下复评」加权项（把 `gate/cost_curve` 的事后检查**提前到搜索回路内**）；成交次数低于门槛的候选视为稀疏有效，不参与排名（评估层 `backtest.py` 已有 `min_trades` 门槛，此处是搜索层补齐） | 快照单测：成本敏感项存在于代统计；稀疏候选被降权 |
| **S6** | **token 剪枝反馈回路** | 按代统计特征/算子 token 被选中频率，低频且低贡献的 token 动态降低采样先验（落点 `rpn.py` `random_feasible` 的先验参数；搜索空间收缩，预算等效放大） | 统计单测：剪枝后 vocab 有效规模下降、最优分不回退 |

工作量：S1–S2 约 2 天；S3–S4 约 3 天；S5–S6 约 2 天；**另预留 1 天基线重冻结**——
S3~S6 改变搜索动力学，同 seed 搜索轨迹全变，`tests/parity` 与 `tests/regression`
的冻结基准必须重跑并重新冻结（基线文件与生成代码版本绑定）。
实施顺序即编号顺序；S1/S2 优先——它们直接决定长任务能不能安心挂机。

## 7. Samples（关键样例代码）

> 均为**新增文件**骨架或既有文件的**增量改动示意**，演示接口形状与工程约定
> （错误码、类型标注、纯度边界），非最终实现。落地时以测试驱动补全。

### Sample A — `Semantics` IR 追加字段（三处同步，缺一即丢字段或漂移 spec_id）

```python
# src/miaosuan/ir/schema.py —— Semantics 追加字段的完整改动面
@dataclass(frozen=True)
class Semantics:
    position_fn: str = "tanh"
    neutral_band: float = DEFAULT_NEUTRAL_BAND
    long_short: bool = True
    warmup_bars: int = DEFAULT_WARMUP_BARS
    timeframe: str = "H1"
    roll_window: int = 500
    # ── v2 追加：运行时口径（旧 spec 反序列化取默认值）──
    execution_lag_bars: int = 1   # 信号 bar 与可执行 bar 的延迟；与回测执行口径
                                  # target_ret[t] = log(open[t+2]/open[t+1]) 一致
                                  # （bar t 收盘出信号 → t+1 开盘执行），回测/纸面/实盘共用

    def to_dict(self) -> dict[str, Any]:
        return {
            # ...既有 6 键逐项不动...
            "execution_lag_bars": self.execution_lag_bars,   # ← ① 必须同步加，否则新 spec 落盘丢字段
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Semantics":
        return cls(
            # ...既有 6 字段逐项不动...
            execution_lag_bars=int(data.get("execution_lag_bars", 1)),   # ← ② 旧 spec 缺该键 → 默认值
        )
```

> 兼容与身份策略：
>
> - 反序列化兼容由 `schema.py` 里**逐字段手写**的 `from_dict`/`to_dict` 保证——
>   `codec.loads` 只是转调 `StrategySpec.from_dict`，本身不做字段默认值。
>   ①②两处各加一行，缺任何一处都会静默丢字段（实时/回测口径随之漂移）。
> - **spec_id 漂移**：`spec_id` 哈希含 `semantics.to_dict()`。字段一旦进入
>   `to_dict`，所有旧 spec 重算出的 spec_id 全部变化（即使语义等价）。
>   处理二选一：**(a) 推荐**——`_stable_hash` 的核心字典剔除纯运行时口径字段
>   （spec_id 表达"策略身份"，执行口径变更不应换身份）；(b) 全量重算并公告迁移，
>   逐一核对指纹/门禁记录/regression 基线中的 spec_id 引用。落地时二选一写进 PR 描述。
> - `neutral_band` 同时是实时侧观望阈值（`|position| < neutral_band` 不交易），
>   此前由回测/预览路径消费（cli.py 映射到 `MIN_TRADE_EXPOSURE`，两者同为 0.05），
>   v2 起三方（回测/纸面/实盘）共用同一字段，禁止另立配置。
> - `Semantics` 若再变更，必须走 `PROVENANCE` 记录 + 对拍测试升级，禁止静默改默认值。

### Sample B — runtime 端口定义（`src/miaosuan/runtime/ports.py`）

```python
# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
"""运行时端口：数据 / 执行 / 通知 三个可插拔抽象。

依赖方向约束：本模块只 import 标准库；
具体实现（mt5/paper/feishu/shenji…）位于 ``runtime/brokers|notifiers|sources``，
运行时通过工厂注入，**不感知具体实现**。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass


@dataclass(frozen=True)
class Bar:
    symbol: str
    timeframe: str
    ts: int          # UTC epoch 秒，bar 开盘时间（契约同 panel.py）
    open: float
    high: float
    low: float
    close: float
    volume: float
    closed: bool     # 必须为 True 才允许进入信号引擎（收盘纪律在端口层强制）


class DataPort(ABC):
    @abstractmethod
    def stream_closed_bars(self, symbol: str, timeframe: str) -> Iterator[Bar]:
        """按周期收盘逐根推送已收盘 Bar；重连后从账本水位补拉，不重不漏。"""

    @abstractmethod
    def symbol_spec(self, symbol: str) -> "BrokerSpec":
        """返回经纪商品种规格（tick_value / lot_step / min_lot / contract_size）。

        BrokerSpec 定义见 market/profiles.py（R1.3），是画像的可选伴随字段。
        """


class ExecutionPort(ABC):
    @abstractmethod
    def submit(self, symbol: str, side: str, lots: float,
               *, magic: int, comment: str = "") -> str:
        """开仓，返回 broker_order_id。实现必须快速返回或内部排队，不得阻塞调度循环。"""

    @abstractmethod
    def close_position(self, position_id: str, *, reason: str) -> None:
        """按持仓 id 平仓。

        MT5 hedging 账户（散户默认）下反向 ``submit`` 是**新开仓**而非平仓；
        平仓必须携带持仓 ticket（请求体 ``position`` 字段）。禁止用 submit 平仓。
        """

    @abstractmethod
    def positions(self) -> list[dict]:
        """当前持仓快照（方向/手数/开仓价/magic/ticket），供对账。"""

    @abstractmethod
    def fills(self, since_ts: int) -> list[dict]:
        """``since_ts`` 之后的成交回执，对账器（R3.5）的输入。

        Paper 实现读本地账本；MT5 实现读 ``history_deals_get``。
        没有回执通道就无法逐笔对账。
        """

    @abstractmethod
    def flatten_all(self, *, reason: str) -> None:
        """紧急全平（kill 开关/熔断路径），实现必须尽力而为并如实回报。"""


class NotifierPort(ABC):
    @abstractmethod
    def notify(self, title: str, body: str, *, level: str = "info") -> None: ...
```

### Sample C — PaperBroker（事件式撮合与成本口径的锚点）

```python
# src/miaosuan/runtime/brokers/paper.py（骨架）
class PaperBroker(ExecutionPort):
    """事件式撮合：submit 只入队，下一根已收盘 bar 的开盘价成交。

    - 成本口径（单一事实）：``profile.cost_model``（CostModel：commission /
      slippage / sell_tax，单边比率，买卖可不对称）——画像层不另设费用字段，
      禁止本文件出现任何硬编码费率，这是对拍成立的前提；
    - 经纪商规格：``profile.broker_spec``（lot_step / min_lot / contract_size / tick_value）；
    - 时序：submit 时刻下一根 bar 尚不存在，"下一根开盘价"拿不到值，
      故只能入队（返回 intent_id），由调度器在每根 bar 开盘时回调 ``on_bar_open``；
    - ``flatten_all`` 是急停路径的**明示例外**：按当前 mark 价即时成交，不等下一根开盘。
    """

    def __init__(self, profile: FrozenMarketProfile, ledger: Ledger) -> None:
        self._cost = profile.cost_model
        self._spec = profile.broker_spec
        self._ledger = ledger
        self._pending: list[Intent] = []

    def submit(self, symbol: str, side: str, lots: float, *, magic: int,
               comment: str = "") -> str:
        intent = Intent(symbol=symbol, side=side, lots=lots,
                        magic=magic, comment=comment)
        self._pending.append(intent)
        return self._ledger.append_intent(intent)      # 立即返回 id（未成交状态）

    def on_bar_open(self, bar: Bar) -> None:           # 调度器每根 bar 开盘回调
        for it in self._pending:
            price = bar.open * (1 + self._cost.slippage) if it.side == "buy" \
                else bar.open * (1 - self._cost.slippage)
            fee = price * it.lots * self._spec.contract_size * self._cost.commission
            self._ledger.append_fill(it, price, fee)   # intent → fill，账本落盘
        self._pending.clear()

    def positions(self) -> list[dict]:
        return self._ledger.open_positions()

    def close_position(self, position_id: str, *, reason: str) -> None:
        self._ledger.close(position_id, mark_price=self._mark,
                           comment=f"close:{reason}")   # 账本实现从略

    def flatten_all(self, *, reason: str) -> None:
        for pos in self.positions():
            # 急停例外：即时 mark 价成交（非下一根开盘），账本中明示 reason
            self._ledger.append_fill(pos["symbol"], _opposite(pos["side"]),
                                     pos["lots"], self._mark(pos["symbol"]),
                                     0.0, magic=pos["magic"],
                                     comment=f"flatten:{reason}")
```

### Sample D — MT5Broker 骨架（可选依赖 + 错误码）

```python
# src/miaosuan/runtime/brokers/mt5.py（骨架）
import os

from ...errors import MiaoSuanError

class EBrokerUnavailable(MiaoSuanError):
    code = "E-BROKER-UNAVAILABLE"

def _load_mt5():
    try:
        import MetaTrader5 as mt5          # 仅 [live] extra 提供
    except ImportError as exc:
        raise EBrokerUnavailable(
            "缺少 MetaTrader5：pip install 'miaosuan[live]'（仅 Windows）"
        ) from exc
    if not mt5.initialize():
        raise EBrokerUnavailable(f"MT5 终端初始化失败: {mt5.last_error()}")
    return mt5

class MT5Broker(ExecutionPort):
    def __init__(self) -> None:
        # 凭证（MT5_LOGIN / MT5_PASSWORD / MT5_SERVER）一律 os.environ
        # （gitleaks 红线），禁止硬编码/落盘
        self._mt5 = _load_mt5()

    def submit(self, symbol, side, lots, *, magic, comment=""):
        tick = self._mt5.symbol_info_tick(symbol)
        req = {
            "action": self._mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lots,
            "type": self._mt5.ORDER_TYPE_BUY if side == "buy"
                    else self._mt5.ORDER_TYPE_SELL,
            "price": tick.ask if side == "buy" else tick.bid,  # 市价单参考价
            "deviation": 20,                                   # 允许滑点（点）
            "magic": magic,
            "comment": comment[:31],       # MT5 注释长度上限
            "type_time": self._mt5.ORDER_TIME_GTC,
            "type_filling": self._filling_mode(symbol),
        }
        r = self._mt5.order_send(req)
        if r is None or r.retcode != self._mt5.TRADE_RETCODE_DONE:
            raise EBrokerUnavailable(f"order_send 失败: {getattr(r, 'retcode', None)}")
        return str(r.order)

    def close_position(self, position_id: str, *, reason: str) -> None:
        # hedging 账户平仓：请求体必须带 "position": ticket、方向取反，其余同 submit
        ...  # 落地时以模拟户实测为准（R4.5 验收清单）
```

### Sample E — VolTargetPolicy（纯函数，可注入）

```python
# src/miaosuan/runtime/risk.py（核心逻辑，纯 numpy 可单测）
def vol_target_lots(
    *,
    ref_lots: float,            # 基准手数（RiskPolicy 配置；单品种路线下即本品种基准）
    ref_dollar_vol: float,      # 基准单 bar 美元波动（ATR×contract_size×lots）
    dollar_vol_per_lot: float,  # 目标品种单手单 bar 美元波动
    sharpe_weight: float,       # clamp 到 [w_min, w_max]（RiskPolicy 配置）
    lot_step: float,            # 来自 BrokerSpec
    min_lot: float,             # 来自 BrokerSpec
) -> float:
    """手数 = 基准波动预算 / 目标品种单手波动 × Sharpe 权重，向 lot_step 取整。

    风险红线：``raw < min_lot`` 一律返回 0（不交易）——**禁止**向上取整到
    min_lot：那会让弱信号被迫开出最小仓，放大而非缩小风险。此边界必须有单测。
    """
    raw = ref_lots * (ref_dollar_vol / dollar_vol_per_lot) * sharpe_weight
    if raw < min_lot:
        return 0.0
    return round(math.floor(raw / lot_step) * lot_step, 8)
```

### Sample F — 对拍 harness（闭环正确性的验收基石）

```python
# tests/runtime/test_runtime_signal_parity.py（骨架）
# 注意：tests/parity/test_signal_parity.py 已存在（研究侧对拍），勿同名混淆。
def test_runtime_signal_matches_backtest_bar_by_bar(spec_fixture, recorded_bars):
    """同一份已收盘 bars：回测仓位序列 == 实时信号仓位序列（逐元素相等）。

    两侧唯一允许的公共来源：core.vm + Semantics；
    任何一侧私自引入『未收盘 bar / 不同 lag / 不同中性带』都会在此变红。
    """
    bt_positions = backtest_positions(spec_fixture, recorded_bars)      # 研究侧
    rt = SignalEngine(spec_fixture, data_port=FakeDataPort(recorded_bars))
    rt_positions = [p for p in rt.replay_closed_bars()]                 # 运行时侧
    assert rt_positions == pytest.approx(bt_positions, abs=0.0)         # 逐位一致
```

### Sample G — CLI 接线与启动安全链

```bash
# 数据（R1）
miaosuan fetch --source mt5 --symbol XAUUSD --timeframe M30
# 信号模式（R2，不下单）
miaosuan watch add XAUUSD --spec artifacts/best_XAUUSD.json --profile FOREX_XAUUSD_MT5
# 纸面（R3）
miaosuan paper start --profile FOREX_XAUUSD_MT5 --specs artifacts/best_*.json
# 实盘（R4）：门禁 + 显式确认双保险
miaosuan live start --profile FOREX_XAUUSD_MT5 --specs artifacts/best_*.json \
    --i-understand-real-money
```

```python
# runtime 启动安全链（signal_engine.load_spec 内，顺序不可调换）
# 依赖白名单：core.vocab（词表校验）与 gate（只读 verdict）均在 §3 铁律白名单内
def load_spec_checked(path: Path) -> StrategySpec:
    spec = ir.load(path)
    if spec.provenance.gate_verdict != "DEPLOYABLE":
        raise ERuntimeNotDeployable(spec.provenance.gate_verdict)   # E-RUNTIME-NOT-DEPLOYABLE
    FORMULA_VOCAB.verify(spec.payload.vocab_version)                # E-VOCAB-MISMATCH
    if kill_file_exists():
        raise EBrokerKillswitch("检测到 STOP 文件，拒绝启动")
    return spec
```

### Sample H — pyproject extras

```toml
# pyproject.toml —— 仅新增 live 一行；既有 tune/oracle/web/datasource/dev extras 一律不动
# （v1.0 样例曾把 tune 误写为 optuna>=3.6，实际为 >=3.4，v1.1 起样例不再复写既有行）
[project.optional-dependencies]
live = ["MetaTrader5>=5.0.45 ; sys_platform == 'win32'"]
```

### Sample I — 搜索检查点与分数下限（S1/S2 骨架）

```python
# src/miaosuan/search/ckpt.py（骨架）
@dataclass
class MineCheckpoint:
    generation: int
    population_tokens: list[list[int]]
    elite_pool: list[tuple[float, int, list[int]]]   # (score, birth_gen, tokens)
    rng_state: dict            # numpy Generator 状态，保证恢复后可复现
    budget_state: dict         # 三档预算已消耗量
    best_score: float

def save_ckpt(ckpt: MineCheckpoint, path: Path) -> None:
    """原子落盘。两个坑（v1.0 样例各踩了一个，v1.1 已修）：

    1. ``np.savez`` 对不以 ``.npz`` 结尾的文件名会**静默追加 .npz**，
       后续替换就会找不到文件——必须传文件句柄固定落盘名；
    2. ``rng_state`` / 嵌套结构是对象数组，读回必须 ``allow_pickle=True``。
    """
    tmp = path.with_name(path.stem + ".tmp.npz")
    with open(tmp, "wb") as f:
        np.savez(f, **asdict(ckpt))
    os.replace(tmp, path)                      # 同目录原子替换，防半截文件

def load_ckpt(path: Path) -> MineCheckpoint:
    raw = np.load(path, allow_pickle=True)     # 对象字段需 allow_pickle
    return MineCheckpoint(
        generation=int(raw["generation"]),
        population_tokens=raw["population_tokens"].tolist(),
        elite_pool=[(float(s), int(g), list(t)) for s, g, t in raw["elite_pool"]],
        rng_state=raw["rng_state"].item(),
        budget_state=raw["budget_state"].item(),
        best_score=float(raw["best_score"]),
    )

def floor_protect(candidates: list[Candidate], floor: float) -> list[Candidate]:
    """分数下限：低于 floor 的候选不落盘（弱结果永不覆盖既有冠军）。

    floor 来源：artifacts 中同标的既有最优 spec 的 ``evidence.val_score``。
    """
    kept = [c for c in candidates if c.val_score > floor]
    log.info("floor_protect", kept=len(kept), dropped=len(candidates) - len(kept))
    return kept
```

## 8. 风险与对策

| 风险 | 等级 | 对策 |
|---|---|---|
| 实时/回测口径漂移（未收盘 bar、lag、中性带不一致） | **高** | D2 全部口径进 `Semantics`；Sample F 对拍进 CI，任何一侧改动未过对拍不得合并 |
| IR 加字段导致 spec_id 漂移（指纹/门禁/基线引用静默失配） | 中 | Sample A 方案 (a)：运行时口径字段剔除出 spec_id 哈希；变更前后跑 spec_id 稳定性单测 |
| S3~S6 改变搜索动力学，冻结基线失效（同 seed 轨迹全变） | 中 | 合并前重跑受影响基线并重新冻结；基线文件与生成代码版本绑定（S 系列工作量已含 1 天） |
| MetaTrader5 包仅 Windows / 终端必须登录 | 中 | `[live]` extra 隔离；runtime 无 MT5 时降级信号/纸面模式并明示（`E-BROKER-UNAVAILABLE`） |
| 经纪商时间戳/时区坑（DST、服务器时区） | 中 | 复用 `mt4_bridge.py` 的偏移探测方案；对拍测试用真实录制 bar 兜底 |
| 进程崩溃/断电导致账本与券商状态不一致 | **高** | 账本原子写 + 启动对账先行，不一致先平后报（`E-BROKER-RECONCILE`） |
| hedging 账户误用反向 submit"平仓"导致叠仓 | **高** | ExecutionPort 强制 `close_position(ticket)` 语义（Sample B）；R4.5 模拟户验收含开/平全流程 |
| 实盘代码腐化反噬研究侧纯度 | 中 | R4 合并前 CI 增加 `core/` 纯度检查（G6）；runtime 只经 ports 依赖外部 |
| 文档/命名自动化事故重演（全局替换污染） | 低 | 本文档落地时逐段 review；新增命名走 pre-commit 钩子 |

## 9. 度量与验收总表

| 阶段 | 交付物 | 对应目标 | 完成判据 |
|---|---|---|---|
| R1 | `mt5_src.py` + `FOREX_XAUUSD_MT5` 画像（含 `broker_spec`）+ `[live]` extra | G1 | 集成测试三断言全绿 |
| R2 | `runtime/` 骨架 + 对拍 harness + watchlist | G2 | 对拍 CI 常绿；24h 无漏拍 |
| R3 | PaperBroker + VolTargetPolicy + 账本/对账/kill | G3 | 两周纸面零对账差异 |
| R4 | MT5Broker + 熔断 + 启动安全链 | G4 | 模拟户全流程清单通过 |
| R5 | webui 双通道 + 飞书通知 | G5 | e2e 冒烟通过 |
| 全程 | `core/` diff 为空 | G6 | CI 纯度检查通过 |
| S1–S2 | 检查点续挖 + 分数下限 | G7 | kill/任务管理器强杀恢复测试；下限保护单测 |
| S3–S4 | 精英龄期回放 + 相关性惩罚 | G7 / 搜索质量 | 衰减采样分布单测；泄漏单测 |
| S5–S6 | 代内成本压力 + token 剪枝 | G7 / 搜索质量与预算 | 成本项快照单测；剪枝后最优分不回退 |

## 10. 与现有模块的关系

| 模块 | 动作 | 说明 |
|---|---|---|
| `core/` | **不动** | 硬约束；G6 用 CI 守护 |
| `ir/schema.py` | 追加字段 | `Semantics.execution_lag_bars`（Sample A：`to_dict`/`from_dict`/spec_id 三处同步） |
| `data/fetchers/` | 新增 `mt5_src.py` | 复用 `acquisition` 校验/缓存/增量判定 |
| `market/profiles.py` | 新增 `FOREX_XAUUSD_MT5` 画像 + 可选 `broker_spec` 字段 | 成本仍走 `cost_model` 单一事实；追加可选字段时同步修正 `MARKET_PROFILE_REQUIRED_FIELDS` 的 `[:-1]` 切片假设 |
| `adapters/` | 不动 | `TargetPort`（导出）与本路线 ports 职责正交 |
| `gate/` | 不动，被 runtime 消费 | `verdict == DEPLOYABLE` 成为启动硬门 |
| `webui/` | 扩展实时页 | 本地 runtime / 神机双数据后端 |
| `search/ga.py`、`search/islands.py`、`search/rpn.py` | 扩展（S 系列） | 精英龄期回放、相关性惩罚、成本压力进适应度；token 采样先验落点 `rpn.py` `random_feasible`；检查点状态机加在 `mine` 外层 |
| `pyproject.toml` | 新增 `[live]` extra | 只加一行，既有 extras 不动 |

## 11. 附录 A — 原版 RL miner（M5）功能对照

> 背景：研究侧现实现是对原版 torch/RL miner（冻结基准，`tests/parity` 的 `m5_*` fixtures）的 numpy/GA 重写。
> 本表逐项回答"M5 的 17 项机制在妙算 / 本路线里是否有着落"：
> **已有** = 现实现已覆盖（增量路线图因此不重复展开）；**S 系列** = 本路线补齐；
> **不适用** = RL 特有、GA 架构无对应物（给出等价机制）。

| # | M5 机制 | 状态 | 妙算落点 / 说明 |
|---|---|---|---|
| 1 | 栈深掩码约束采样 + 可达性剪枝 | 已有 | `rpn.py` `random_feasible`：栈深约束 + 每步可行性剪枝 + `avoid_infection` 有界重试 |
| 2 | 感染模型（恒正/传播/治愈） | 已有 | `core/vm.py` 三类算子判定 + `validate_formula_structure`（感染链阈值以 vm.py 实现为准） |
| 3 | 相邻重复 token 惩罚 | 已有（口径微差） | `ga.py`：相邻相同 token 每满 2 次罚 0.3，train/val 双份扣除；对拍基准一致 |
| 4 | 滚动 WF 5 折 + gap 20 | 已有 | `mine.py` `n_folds=5, gap=20`（与对拍基准一致） |
| 5 | IC 符号门控 + IC 稳定性 | 已有 | `ga.py fold_scores` IC 门控均值；`backtest.py` 时序 IC 稳定性（mean/std） |
| 6 | 六项加权 Reward 合成 | 已有（组成项齐） | `backtest.py` MT5Backtest：Sortino / Calmar / IC / 品种一致性 / 成本压力 / 换手质量 / Beta 中性 / 前后一致性；具体权重以代码为准 |
| 7 | 稀疏交易防刷分 | 已有（评估层）+ S5 | `backtest.py` `min_trades=max(5, eval_bars//100)`；S5 把门槛提前到搜索适应度 |
| 8 | 年化因子从时间戳自动估计 | 已有 | `estimate_periods_per_year`（span-based，`tests/parity` 回归覆盖） |
| 9 | 因子池相关性惩罚（train 切片） | **S4** | top-k 因子池，\|corr\| 超阈值降分；只在 train 折切片计算，防 val 泄漏 |
| 10 | 精英回放（龄期半衰 + 软温度采样） | **S3**（字段已有） | `Individual.birth_gen` 已存在（ga.py）但未使用；S3 启用：半衰衰减 + softmax(T<1) 采样 + 按 tokens 去重，注入当代评估与选择；"反哺梯度"RL 特有，GA 等价 = 回放进选择回路 |
| 11 | 特征/token 剪枝反馈回路 | **S6** | 低频低贡献 token 降采样先验（落点 `rpn.py` `random_feasible`） |
| 12 | 熵自适应系数 + 熵下限损失 | 不适用（RL 特有） | GA 等价物：`min_hamming` 多样性硬约束 + 停滞移民（连续无改进注入随机移民）——探索压力永不归零 |
| 13 | 三模式重启（全重置/部分扰动/全扰动） | 不适用（RL 特有） | GA 无参数吸引子；等价：多岛 + patience 移民逃离局部最优 |
| 14 | LoRD 正则 / StableRank / EMA reward 基线 | 不适用（纯 torch/RL） | GA 无梯度参数；torch 仅 `oracle` extra 作回归基准 |
| 15 | 孤岛模型 + 全局最优快照同步 | 已有（快照部分由 S1 承担） | `islands.py`：分区 / 迁移（纯数据搬运，不重复评估）/ 全局最优查询（`best()`）/ 多样性度量；无独立快照同步机制，S1 检查点落盘 `best_score` 覆盖恢复需求 |
| 16 | 分数下限 + 检查点 + 恢复时精英回放 | **S1 + S2 + S3** | 检查点原子落盘（Sample I）；下限来源 = 既有 best spec 的 `evidence.val_score`；恢复后先执行一次精英回放 |
| 17 | 搜索健康监控（熵/KL/有效词汇/多样性） | 部分已有 | `ga.py` 日志：gen / best / diversity / evals / immigrants；熵/KL 不适用；有效词汇规模 → S6 验收指标 |

---

*后续修订：本文按 R1→R5 与 S1→S6 推进滚动更新；每阶段合并时在本表打勾并附 PR 链接。*
