# 数据适配坑清单（T02 输入）

> 状态：**记录，未实现**。本文档由 M5（T01 收官）沉淀，用于指导 T02 的 `data/` 层实现。
> 所有条目均来自对冻结 AlphaMaster 数据路径的**逐行核对**与端到端对拍（XAUUSD_H1）实测，
> 而非臆测。妙算 `core/` **不做任何数据 IO / 适配**；适配逻辑一律下沉到 T02 的 `data/`。
>
> 预留接口见 `src/miaosuan/core/ports.py`（`DataSource` / `MarketProfile` / `CostModelProtocol`）。

数据源文件（对拍用）：`D:\K线数据\XAUUSD_H1.parquet`（8000 根 H1）。
基准事实（`tests/fixtures/e2e_xauusd_meta.json` → `data_facts`）：
`rows_raw = rows_used = 8000`，列为 `[time, open, high, low, close, volume]`，
`time ∈ [1746442800, 1788962400]`，`has_tick_volume = false`。

---

## 1. 复权 / 价格调整（adjustment）

- **现状**：AlphaMaster 的 `ParquetDataManager` **不做任何复权**：直接取列原始 OHLC，转
  `float32`。它假设 parquet 已是「目标口径」的价格序列（期货连续合约 / 现货 / 已复权股票）。
- **坑**：股票品种若为「不复权 / 前复权 / 后复权」不同口径，因子数值与回测收益会系统性
  偏差；期货主力连续合约存在**换月跳空**，会污染 `MA_DIFF` / `MOMENTUM` / `TRIX` 等差分特征。
- **T02 动作**：
  - `DataSource.adjustment_mode` 显式声明口径（`none | qfq | hfq | backadj`），**默认 `none`
    以对齐 AM**；
  - 复权在**加载后、特征前**统一施加，禁止只改部分字段；
  - 期货连续合约需支持「价差调整 / 比例调整」开关，并在 `MarketProfile` 标注 `is_continuous`。
- **对拍影响**：端到端对拍用 `none` 口径，故与 AM 逐位一致；任何复权模式都是**新语义**，
  必须单独建对拍基准，不得混用。

## 2. 时间戳单位修复（`time` 被误存为「秒/1000」）

- **AM 语义**（`parquet_manager.py`）：
  ```python
  if is_numeric_dtype(time) and time.max() < 10_000_000:   # 1970-04-27 之前
      time = time * 1000                                   # 视为被除过 1000，乘回
  ```
- **坑**：部分 A 股导出工具把 Unix **秒**误存为「秒/1000」，日期会退到 1970 年，导致跨品种
  时间轴对齐完全错乱。
- **T02 动作**：保留该修复但**升级为显式列契约**：
  - 声明 `time_unit ∈ {s, ms}`，默认**自动探测**（阈值 `1e7` 与 AM 一致）；
  - 探测结果写入加载日志/元信息，便于审计；
  - 拒绝「非单调可修复」的时间戳（如出现负数、NaN）。
- **风险**：阈值法在「极早期历史数据」上可能误判；T02 建议以「中位数时间戳落在合理年份区间」
  做二次校验。

## 3. 缺失 bar / 重复时间戳（去重与排序）

- **AM 语义**：
  ```python
  sub = sub.sort_values("time")
  sub = sub[~sub["time"].duplicated(keep="last")]   # 重复时间戳保留最后一条
  ```
- **坑**：
  - 同一时间戳多次导出（拼接/追加写）→ 必须去重，否则时间轴出现「重复 K 线」；
  - 排序缺失（文件乱序）→ 差分/滚动特征全错；
  - **`keep="last"` 是明确选择**（保留最新写入），与 `keep="first"` 结果不同，必须对齐。
- **T02 动作**：`DataSource.load()` 内固定「按 `time` 升序 + `duplicated(keep="last")`」，
  并统计 `n_dropped_duplicates` 写入元信息。
- **对拍**：端到端测试 `test_e2e_data_path` 已断言重建面板与 AM 基准**逐位一致**，
  含去重/排序语义。

## 4. 多品种时间轴对齐（alignment）

- **AM 语义**（`MT5DataManager._align_timelines`）：
  1. **优先取时间戳交集**（只保留所有品种都有真实报价的 bar），以彻底消除休市
     `forward-fill` 造成的「伪重复 K 线」；
  2. 交集不足 `MIN_BARS` 时**降级为并集 + `ffill`** 并告警；
  3. **只允许 `ffill`（因果填充），严禁 `bfill`**（避免未来信息泄漏）；
  4. 起始处缺历史 → `fillna(0.0)`（避免下游 `log` / `divide` 出 `-inf`）。
- **坑**：并集 + 双向填充是最常见的 look-ahead 泄漏源；`fillna(0.0)` 会把「无报价」当成
  「价格为 0」，对**价格类**特征是隐式污染（AM 接受此权衡）。
- **T02 动作**：
  - `DataSource.trading_calendar` / `tradable_mask` 显式表达「该 bar 是否有真实报价」；
  - 对齐策略可配置（`intersection | union_ffill`），默认对齐 AM（intersection 优先）；
  - 在特征计算前把「无报价」的 bar 标记出来，供特征层决定 `NaN` 还是 `0`（**不要把 0 当价格**）。
- **单品种 vs 多品种**：`N=1` 走滚动时序归一，`N>1` 走截面归一（见 `core/vm.py`）。对齐会改变
  `N`，从而**切换归一化路径**——对齐策略必须在上游一次性定死，不能中途改变 `N`。

## 5. 成交量列名（`tick_volume` vs `volume`）

- **AM 语义**：
  ```python
  volume_col = "tick_volume" if "tick_volume" in df.columns else "volume"
  ```
- **坑**：MT5 导出为 `tick_volume`，第三方导出为 `volume`；若列选择错误，`volume` 相关特征
  （`VWAP` / `VWAP_DEV` / 量价类）全错或抛 `KeyError`。
- **T02 动作**：`MarketProfile.volume_semantics ∈ {tick, real}`；列解析按 AM 优先级，并在
  元信息记录实际使用的列名。对拍数据为 `volume`（`has_tick_volume = false`）。

## 6. 目标收益 `target_ret`（前视 horizon = 2）

- **AM 语义**（`MT5DataManager._compute_target_ret`）：
  ```
  target_ret[n, t] = log(open[n, t+2] / open[n, t+1])   # t ∈ [0, T-3]
  target_ret[n, T-2] = target_ret[n, T-1] = 0           # 末两根零填充（边界）
  safe_denom = open[t+1]; safe_denom[safe_denom == 0] = 1.0   # 防除零
  ```
- **坑**：
  - 用**开盘价**而非收盘价，且是 `t+1 → t+2` 而非 `t → t+1`（与「次日开盘进场」的交易语义对齐）；
  - 末两根**必须零填充**，否则 `_align_causal`（裁掉末尾 horizon=2 步）会把边界错位；
  - 分母为 0 的防御处理必须保留（否则出 `inf`）。
- **T02 动作**：`TargetPort` 暴露 `compute_target_ret(open, horizon=2)`，语义与 AM 逐点一致；
  `horizon` 与 `EffectivenessEvaluator.target_horizon` 必须**同源**（默认均为 2）。
- **对拍口径**：评估器 `_align_causal` 与特征层 warm-up 必须一致到同一时间基准
  （见 `core/evaluator.py` / `core/vm.py`）。

## 7. 年化期数 `periods_per_year`（按周期）

- **AM 语义**：H1 固定用 **6240**（24h 市场，`MT5Backtest(periods_per_year=6240)`）；
  另可从实际时间跨度估算年数（`inspect_parquet_file`：`span_seconds / (365.25*24*3600)`）。
- **坑**：年内期数错 → Sharpe/Sortino/Calmar 年化系数错，回测打分失真；不同周期（M1/M5/…/D1）
  期数不同。
- **T02 动作**：`MarketProfile.periods_per_year` 由 `timeframe` 推导（H1 → 6240），
  并允许用真实时间跨度**覆盖**（优先实测）。`estimate_periods_per_year` 已在
  `core/backtest.py` 提供。

## 8. 其它工程约束

| 项 | AM 语义 | 妙算约定 |
| --- | --- | --- |
| 最小 bar 数 | `Config.MIN_BARS`（不足则排除品种） | T02 显式参数，默认对齐 AM |
| dtype | OHLCV → `float32` | 一致（float32），避免精度漂移 |
| 文件名契约 | `{symbol}_{timeframe}.parquet`（支持别名） | T02 复用 `parse_parquet_filename` 语义 |
| 列集合 | `time, open, high, low, close, volume` | `DataSource.load()` 返回同构 `raw_dict` |
| 缺失列 | `raise ValueError(缺列)` | 一致：显式报错，不静默补列 |

---

## 9. 端到端对拍实测结论（M5）

- 数据路径：妙算独立重建 `raw_dict` 与 AM 基准**逐位一致**（`test_e2e_data_path` 通过）。
- 特征层：65 维逐点 `max|Δ| ≈ 2.8e-2`（主要来自 `TRIX_SIGNAL` / `TRIX_15` 零点附近的
  float32 归约噪声）；`|Δ|>1e-2` 的占比 ≤ 1e-3。
- 因子层：`p99.9|Δ| ≈ 4.3e-3`；`|Δ|>0.1` 仅 **1** 根 bar（`t=6231`），系 `GATE` 条件
  （`feat33 = TRIX_15`，`0.0` vs `+9.4e-4`）的**双线性不连续**（1-ULP 符号翻转），
  非移植 bug；剔除近零条件 bar 后 `max|Δ| ≈ 4.4e-3`。
- 聚合统计（妙算 vs AM）：`mean 0.258978 vs 0.258959`、`std 0.811758 vs 0.811755`、
  `max 3.0 = 3.0`、在市占比 `0.920625 = 0.920625`。

> **给 T02 的提醒**：任何复权 / 对齐策略的改变都会**同时改变特征与因子**，
> 必须重建独立对拍基准；不要把「上游数据口径差异」误判为「移植数值误差」。
