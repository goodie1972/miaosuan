import sys, pandas as pd, numpy as np
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, 'src')
cache_dir = Path(r'D:\backup\BaoBao\PythonProgram\miaosuan\data\cache')

files = [
    'XAUUSD_H1.parquet',
    'XAUUSD_H1_network.parquet',
    'XAUUSD_H1_Dukascopy_full.parquet',
]

lines = []
lines.append(f"{'文件':<40} {'行数':>10} {'最早':>20} {'最新':>20} {'大小(MB)':>8}")
lines.append("-" * 105)

dfs = {}
for fname in files:
    f = cache_dir / fname
    df = pd.read_parquet(f)
    dfs[fname] = df
    min_ts = int(df['time'].min())
    max_ts = int(df['time'].max())
    min_dt = datetime.fromtimestamp(min_ts, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')
    max_dt = datetime.fromtimestamp(max_ts, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')
    size_mb = f.stat().st_size / 1e6
    lines.append(f"{fname:<40} {len(df):>10} {min_dt:>20} {max_dt:>20} {size_mb:>8.2f}")

lines.append("\n=== H1(原) vs Dukascopy_full 对比 ===")
h1_orig = dfs['XAUUSD_H1.parquet']
h1_full = dfs['XAUUSD_H1_Dukascopy_full.parquet']

merged = pd.merge(h1_orig, h1_full, on='time', how='outer', suffixes=('_orig', '_full'), indicator=True)
lines.append(f"合并后总行数: {len(merged)}")
lines.append(f"  仅在 orig: {(merged['_merge']=='left_only').sum()}")
lines.append(f"  仅在 full: {(merged['_merge']=='right_only').sum()}")
lines.append(f"  两者都有: {(merged['_merge']=='both').sum()}")

both = merged[merged['_merge']=='both'].copy()
for col in ['open', 'high', 'low', 'close', 'volume']:
    diff = (both[f'{col}_orig'] - both[f'{col}_full']).abs()
    max_diff = diff.max()
    mean_diff = diff.mean()
    pct_diff = (diff / both[f'{col}_full'].abs().replace(0, np.nan) * 100).dropna()
    lines.append(f"  {col}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, max_pct={pct_diff.max():.4f}%")

orig_only = merged[merged['_merge']=='left_only']['time']
if len(orig_only) > 0:
    lines.append(f"\n仅在 orig ({len(orig_only)} 个，最新 20 个):")
    for ts in sorted(orig_only)[-20:]:
        dt = datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime('%Y-%m-%d %H:%M')
        row = h1_orig[h1_orig['time']==ts].iloc[0]
        lines.append(f"  {dt}: O={row['open']:.3f} H={row['high']:.3f} L={row['low']:.3f} C={row['close']:.3f} V={row['volume']:.2f}")

full_only = merged[merged['_merge']=='right_only']['time']
if len(full_only) > 0:
    lines.append(f"\n仅在 full ({len(full_only)} 个，最早 20 个):")
    for ts in sorted(full_only)[:20]:
        dt = datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime('%Y-%m-%d %H:%M')
        row = h1_full[h1_full['time']==ts].iloc[0]
        lines.append(f"  {dt}: O={row['open']:.3f} H={row['high']:.3f} L={row['low']:.3f} C={row['close']:.3f} V={row['volume']:.2f}")

lines.append("\n=== network (TV) vs Dukascopy_full 对比 ===")
h1_net = dfs['XAUUSD_H1_network.parquet']
merged3 = pd.merge(h1_net, h1_full, on='time', how='outer', suffixes=('_net', '_full'), indicator=True)
lines.append(f"合并后总行数: {len(merged3)}")
lines.append(f"  仅在 net: {(merged3['_merge']=='left_only').sum()}")
lines.append(f"  仅在 full: {(merged3['_merge']=='right_only').sum()}")
lines.append(f"  两者都有: {(merged3['_merge']=='both').sum()}")

both3 = merged3[merged3['_merge']=='both'].copy()
for col in ['open', 'high', 'low', 'close', 'volume']:
    diff = (both3[f'{col}_net'] - both3[f'{col}_full']).abs()
    max_diff = diff.max()
    mean_diff = diff.mean()
    pct_diff = (diff / both3[f'{col}_full'].abs().replace(0, np.nan) * 100).dropna()
    lines.append(f"  {col}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, max_pct={pct_diff.max():.4f}%")

with open('h1_compare_out.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print('Done, check h1_compare_out.txt')