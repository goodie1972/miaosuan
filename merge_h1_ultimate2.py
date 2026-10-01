import sys, pandas as pd, numpy as np
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, 'src')
cache_dir = Path(r'D:\backup\BaoBao\PythonProgram\miaosuan\data\cache')

h1_full = pd.read_parquet(cache_dir / 'XAUUSD_H1_Dukascopy_full.parquet')
h1_orig = pd.read_parquet(cache_dir / 'XAUUSD_H1.parquet')

h1_full = h1_full.sort_values('time').reset_index(drop=True)
h1_orig = h1_orig.sort_values('time').reset_index(drop=True)

full_max_ts = int(h1_full['time'].max())

patch = h1_orig[h1_orig['time'] > full_max_ts].copy()

merged = pd.concat([h1_full, patch], ignore_index=True)
merged = merged.sort_values('time').reset_index(drop=True)
merged = merged[~merged['time'].duplicated(keep='first')].reset_index(drop=True)

cols = ['time', 'open', 'high', 'low', 'close', 'volume', 'tick_volume']
merged = merged[cols]

out_path = cache_dir / 'XAUUSD_H1_ultimate.parquet'
tmp = out_path.with_suffix('.tmp')
merged.to_parquet(tmp, index=False)
tmp.replace(out_path)

diffs = merged['time'].diff().dropna().astype(int)
expected = {3600, 7200, 10800, 14400}
unexpected = diffs[~diffs.isin(expected)]

# Write results to text file
with open('h1_merge_results.txt', 'w', encoding='utf-8') as f:
    f.write(f"基础最新时间: {datetime.fromtimestamp(full_max_ts, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')}\n")
    f.write(f"补丁行数: {len(patch)}\n")
    if len(patch) > 0:
        f.write(f"补丁范围: {datetime.fromtimestamp(int(patch['time'].min()), tz=timezone.utc).strftime('%Y-%m-%d %H:%M')} → {datetime.fromtimestamp(int(patch['time'].max()), tz=timezone.utc).strftime('%Y-%m-%d %H:%M')}\n")
    f.write(f"\n=== 合并完成 ===\n")
    f.write(f"输出文件: {out_path}\n")
    f.write(f"总行数: {len(merged)}\n")
    f.write(f"时间范围: {datetime.fromtimestamp(int(merged['time'].min()), tz=timezone.utc).strftime('%Y-%m-%d %H:%M')} → {datetime.fromtimestamp(int(merged['time'].max()), tz=timezone.utc).strftime('%Y-%m-%d %H:%M')}\n")
    f.write(f"异常间隙: {len(unexpected)} 个\n")
    if len(unexpected) > 0:
        f.write(f"  样例: {unexpected.head(10).tolist()}\n")
    f.write(f"\n与 XAUUSD_H1.parquet 差异: {len(merged) - len(h1_orig)} 行\n")
    f.write(f"与 XAUUSD_H1_Dukascopy_full.parquet 差异: {len(merged) - len(h1_full)} 行\n")

print('Done')