import sys, pandas as pd, numpy as np
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, 'src')
cache_dir = Path(r'D:\backup\BaoBao\PythonProgram\miaosuan\data\cache')

files = [
    'XAUUSD_D1.parquet',
    'XAUUSD_D1_Dukascopy.parquet',
    'XAUUSD_D1_Dukascopy_full.parquet',
    'XAUUSD_D1_network.parquet',
]

print(f"{'文件':<35} {'行数':>8} {'最早':>20} {'最新':>20} {'大小(MB)':>8}")
print("-" * 100)

dfs = {}
for fname in files:
    f = cache_dir / fname
    df = pd.read_parquet(f)
    dfs[fname] = df
    min_ts = int(df['time'].min())
    max_ts = int(df['time'].max())
    min_dt = datetime.fromtimestamp(min_ts, tz=timezone.utc).strftime('%Y-%m-%d')
    max_dt = datetime.fromtimestamp(max_ts, tz=timezone.utc).strftime('%Y-%m-%d')
    size_mb = f.stat().st_size / 1e6
    print(f"{fname:<35} {len(df):>8} {min_dt:>20} {max_dt:>20} {size_mb:>8.2f}")

print("\n=== 逐日对比：Dukascopy_full vs D1 (原) ===")
d1_old = dfs['XAUUSD_D1.parquet']
d1_full = dfs['XAUUSD_D1_Dukascopy_full.parquet']

# 合并对比
merged = pd.merge(d1_old, d1_full, on='time', how='outer', suffixes=('_old', '_full'), indicator=True)
print(f"合并后总行数: {len(merged)}")
print(f"  仅在 old: {(merged['_merge']=='left_only').sum()}")
print(f"  仅在 full: {(merged['_merge']=='right_only').sum()}")
print(f"  两者都有: {(merged['_merge']=='both').sum()}")

# 两者都有时的差异
both = merged[merged['_merge']=='both'].copy()
for col in ['open', 'high', 'low', 'close', 'volume']:
    diff = (both[f'{col}_old'] - both[f'{col}_full']).abs()
    max_diff = diff.max()
    mean_diff = diff.mean()
    pct_diff = (diff / both[f'{col}_full'].abs().replace(0, np.nan) * 100).dropna()
    print(f"  {col}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, max_pct={pct_diff.max():.4f}%")

# 仅在 full 中的日期（新增的）
new_dates = merged[merged['_merge']=='right_only']['time']
if len(new_dates) > 0:
    print(f"\n新增日期 ({len(new_dates)} 个):")
    for ts in sorted(new_dates):
        dt = datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime('%Y-%m-%d')
        row = d1_full[d1_full['time']==ts].iloc[0]
        print(f"  {dt}: O={row['open']:.3f} H={row['high']:.3f} L={row['low']:.3f} C={row['close']:.3f} V={row['volume']:.2f}")

# 仅在 old 中的日期（old 有但 full 无）
old_only = merged[merged['_merge']=='left_only']['time']
if len(old_only) > 0:
    print(f"\n仅在 old ({len(old_only)} 个):")
    for ts in sorted(old_only):
        dt = datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime('%Y-%m-%d')
        row = d1_old[d1_old['time']==ts].iloc[0]
        print(f"  {dt}: O={row['open']:.3f} H={row['high']:.3f} L={row['low']:.3f} C={row['close']:.3f} V={row['volume']:.2f}")

print("\n=== Dukascopy vs Dukascopy_full 对比 ===")
d1_duk = dfs['XAUUSD_D1_Dukascopy.parquet']
merged2 = pd.merge(d1_duk, d1_full, on='time', how='outer', suffixes=('_duk', '_full'), indicator=True)
print(f"合并后总行数: {len(merged2)}")
print(f"  仅在 duk: {(merged2['_merge']=='left_only').sum()}")
print(f"  仅在 full: {(merged2['_merge']=='right_only').sum()}")
print(f"  两者都有: {(merged2['_merge']=='both').sum()}")

both2 = merged2[merged2['_merge']=='both'].copy()
for col in ['open', 'high', 'low', 'close', 'volume']:
    diff = (both2[f'{col}_duk'] - both2[f'{col}_full']).abs()
    max_diff = diff.max()
    print(f"  {col}: max_diff={max_diff:.6f}")

print("\n=== network (TV) vs Dukascopy_full 对比 ===")
d1_net = dfs['XAUUSD_D1_network.parquet']
merged3 = pd.merge(d1_net, d1_full, on='time', how='outer', suffixes=('_net', '_full'), indicator=True)
print(f"合并后总行数: {len(merged3)}")
print(f"  仅在 net: {(merged3['_merge']=='left_only').sum()}")
print(f"  仅在 full: {(merged3['_merge']=='right_only').sum()}")
print(f"  两者都有: {(merged3['_merge']=='both').sum()}")

both3 = merged3[merged3['_merge']=='both'].copy()
for col in ['open', 'high', 'low', 'close', 'volume']:
    diff = (both3[f'{col}_net'] - both3[f'{col}_full']).abs()
    max_diff = diff.max()
    mean_diff = diff.mean()
    pct_diff = (diff / both3[f'{col}_full'].abs().replace(0, np.nan) * 100).dropna()
    print(f"  {col}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, max_pct={pct_diff.max():.4f}%")

# 仅在 net 中的日期
net_only = merged3[merged3['_merge']=='left_only']['time']
if len(net_only) > 0:
    print(f"\nTV独有日期 ({len(net_only)} 个，最新 {len(net_only)} 个):")
    for ts in sorted(net_only)[-10:]:
        dt = datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime('%Y-%m-%d')
        row = d1_net[d1_net['time']==ts].iloc[0]
        print(f"  {dt}: O={row['open']:.3f} H={row['high']:.3f} L={row['low']:.3f} C={row['close']:.3f} V={row['volume']:.2f}")