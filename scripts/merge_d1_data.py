#!/usr/bin/env python3
"""合并 D1 数据：MT5 (2004-2006) + M1聚合 (2006+)"""
import pandas as pd
import numpy as np
from pathlib import Path

cache = Path('data/cache')

print('=== 合并 D1 数据 (MT5 2004-2006 + M1聚合 2006+) ===')

# Load M1-aggregated D1 (2006+)
df_d1_m1 = pd.read_parquet(cache / 'XAUUSD_D1.parquet')
df_d1_m1['time_dt'] = pd.to_datetime(df_d1_m1['time'], unit='s', utc=True)
df_d1_m1 = df_d1_m1.sort_values('time_dt')

# Load MT5 D1 (2004+)
df_d1_mt5 = pd.read_parquet(cache / 'XAUUSD_D1_MT5.parquet')
df_d1_mt5['time_dt'] = pd.to_datetime(df_d1_mt5['time'], unit='s', utc=True)
df_d1_mt5 = df_d1_mt5.sort_values('time_dt')

# Filter MT5 to pre-2006 only
df_d1_mt5_early = df_d1_mt5[df_d1_mt5['time_dt'] < '2006-10-05']
print(f'  MT5 D1 (pre-2006): {len(df_d1_mt5_early):,} rows')

# Filter M1 to >= 2006
df_d1_m1_late = df_d1_m1[df_d1_m1['time_dt'] >= '2006-10-05']
print(f'  M1 D1 (2006+): {len(df_d1_m1_late):,} rows')

# Merge with proper time alignment
df_merged = pd.concat([df_d1_mt5_early, df_d1_m1_late], ignore_index=True)
df_merged = df_merged.sort_values('time_dt').reset_index(drop=True)

# Convert time back to unix timestamp (use numpy int64 to avoid Timestamp.astype issue)
df_merged['time'] = (df_merged['time_dt'].astype('int64') // 10**9).astype('int64')

# Quality check
df_merged['next_open'] = df_merged['open'].shift(-1)
df_merged['gap'] = abs(df_merged['next_open'] - df_merged['close'])
df_merged['dow'] = df_merged['time_dt'].dt.dayofweek

# Weekday gaps (Mon-Fri to Mon-Fri)
weekday_mask = (df_merged['dow'] < 4) & (df_merged['dow'].shift(-1) < 4)
weekday_gaps = df_merged[weekday_mask]['gap']

print(f'  Merged D1: {len(df_merged):,} rows')
print(f'  Time range: {df_merged["time_dt"].iloc[0]} ~ {df_merged["time_dt"].iloc[-1]}')
print(f'  Price range: ${df_merged["low"].min():.2f} ~ ${df_merged["high"].max():.2f}')
print(f'  Weekday gap_mean: ${weekday_gaps.mean():.4f}')
print(f'  Weekday gap>1: {(weekday_gaps>1).mean()*100:.2f}%')

# Save
df_merged[['time', 'open', 'high', 'low', 'close', 'volume']].to_parquet(cache / 'XAUUSD_D1.parquet', index=False)
print('✅ Saved merged D1 data')

# Final check all files
print(f'\n=== 最终缓存文件 ===')
for f in sorted(cache.glob('XAUUSD_*.parquet')):
    if 'MT5' not in f.name:
        df = pd.read_parquet(f)
        print(f'  {f.name}: {len(df):,} rows')