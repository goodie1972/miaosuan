# XAUUSD K线数据溯源与时区对齐指南

## 数据源清单

| 数据源 | 位置 | 时间戳基准 | 实际含义 | 新鲜度 |
|--------|------|-----------|---------|--------|
| kline parquet | `data/cache/XAUUSD_*.parquet` | UTC epoch | **真 UTC** | 至 2026-10-01 |
| ShenjiDB | `D:\backup\...\xauusd\data\market_data.db` | UTC epoch | **UTC+3 服务器标签** | 至 2026-10-22 |
| am_parquet | legacy AM platform | UTC epoch | **UTC+3 服务器标签** | 未知 |
| 标准目录 | `D:\K线数据\XAUUSD_*.parquet` | UTC epoch | **UTC+8** (实测) | 部分缺失 |

## 时区对齐规则

### 跨源比对

```python
# 所有数据现在都是 UTC+3，无需额外转换
# kline parquet: 已转换为 UTC+3
# ShenjiDB: 原始就是 UTC+3 服务器标签

# 直接比对
df_kline = pd.read_parquet('data/cache/XAUUSD_H1.parquet')
df_db = shenji.fetch_full('XAUUSD', 'H1')

# 时间戳已经对齐，可以直接 merge
df_merged = pd.merge(df_kline, df_db, on='time', suffixes=('_kline', '_db'))
```

### 周期桶对齐检查

| 周期 | 桶边界 | 是否已对齐 |
|------|--------|-----------|
| M5/M15/M30/H1 | 3h 移位后自然对齐 | ✓ 已对齐 |
| **H4** | 从 M1 重新聚合，offset=3600 (01:00 UTC+3) | ✓ 已对齐 |
| **D1** | 从 M1 重新聚合，offset=0 (00:00 UTC+3) | ✓ 已对齐 |

### H4 桶原点（已修复）

```
kline H4 (从 M1 重新聚合): 01:00/05:00/09:00/13:00/17:00/21:00 UTC+3
                            time % 14400 == 3600

ShenjiDB H4 (实盘):         01:00/05:00/09:00/13:00/17:00/21:00 UTC+3
                            time % 14400 == 3600

✓ 现在完全一致！
```

### D1 桶原点（已修复）

```
kline D1 (从 M1 重新聚合):  00:00 UTC+3
                            time % 86400 == 0

ShenjiDB D1 (实盘):         00:00 UTC+3
                            time % 86400 == 0

✓ 现在完全一致！
```

## 使用建议

### ✓ 可以混用的场景

1. **所有周期跨源比对**
   - 所有数据都是 UTC+3，无需转换
   - H4/D1 桶边界已与实盘对齐
   - 可直接比较 OHLCV

2. **单源内部分析**
   - kline 内部：直接用 UTC+3
   - DB 内部：直接用 UTC+3
   - 不需要转换

3. **跨周期策略**
   - fish_eaten 类 M30×H4 加权策略：安全
   - sanqing_h1 类策略：安全
   - 所有周期桶边界已统一

### ⚠ 注意事项

1. **价格差异**
   - kline 数据来自 Dukascopy（CFD/forex broker）
   - ShenjiDB 数据来自实盘经纪商
   - 同一时间戳的价格可能有小幅差异（中位差 ~0.25 for H4, ~0.67 for D1）
   - 这是数据源差异，不是桶对齐问题

2. **数据新鲜度**
   - kline M1 截止 2026-10-20（ShenjiDB 只有 5000 根 M1）
   - kline M5/M15/M30/H1/H4/D1 截止 2026-10-23
   - ShenjiDB 数据更新到 2026-10-23

3. **历史价格验证**
   - 2026-03-25 之前的 kline 价格无外部锚
   - 仅内部连续性可查（已查，干净）
   - 价格准确性只能靠行情源交叉核对

## 数据新鲜度

| 数据源 | 截止日期 | 状态 |
|--------|---------|------|
| kline parquet | 2026-10-01 | ⚠ 比 DB 旧 3 周 |
| ShenjiDB | 2026-10-22 | ✓ 最新 |
| 标准目录 | 部分缺失 | ⚠ D1 仅 1,602 根 |

### 建议操作

1. **更新 kline 数据**
   - 从 Dukascopy 拉取 2026-10-01 ~ 当前
   - 或从 ShenjiDB 导出并转换时区

2. **验证价格**
   - 2026-03-25 之前的 kline 价格需要外部锚
   - 建议用 TradingView 或 Dukascopy API 交叉核对

3. **统一数据源**
   - 对于 H4/D1 策略，选择一个数据源
   - 推荐：ShenjiDB（最新，实盘同源）

## 代码示例

### 跨源对齐（所有周期，已统一）

```python
import pandas as pd

# 加载 kline (UTC+3)
df_kline = pd.read_parquet('data/cache/XAUUSD_H4.parquet')

# 加载 ShenjiDB (UTC+3)
from miaosuan.data.fetchers.shenji_db import ShenjiDBFetcher
fetcher = ShenjiDBFetcher('D:/backup/BaoBao/PythonProgram/xauusd/data/market_data.db')
df_db = fetcher.fetch_full('XAUUSD', 'H4')

# 时间戳已经对齐，直接合并
df_merged = pd.merge(
    df_kline[['time', 'close']],
    df_db[['time', 'close']],
    on='time',
    suffixes=('_kline', '_db')
)

# 比较价格
df_merged['diff'] = df_merged['close_kline'] - df_merged['close_db']
print(f"中位差: {df_merged['diff'].median():.4f}")
print(f"完全匹配 (<0.01): {(df_merged['diff'].abs() < 0.01).sum()}")
```

### 跨周期策略（安全使用）

```python
# fish_eaten 类 M30×H4 加权策略
# 现在所有周期桶边界已统一，可以安全使用

df_m30 = pd.read_parquet('data/cache/XAUUSD_M30.parquet')
df_h4 = pd.read_parquet('data/cache/XAUUSD_H4.parquet')

# H4 桶边界: time % 14400 == 3600 (01:00 UTC+3)
# M30 桶边界: time % 1800 == 0 (自然对齐)

# 可以安全地进行跨周期计算
# 例如：用 H4 趋势过滤 M30 信号
```

## 参考

- Panel 合约：`src/miaosuan/data/panel.py`
- 时区转换：`src/miaosuan/data/fetchers/mt4_bridge.py` (line 29)
- 桶对齐：`src/miaosuan/data/fetchers/dukascopy.py` (bar aggregation)
- 数据清理：`scripts/clean_xauusd_data.py`
