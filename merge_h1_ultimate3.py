import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

cache_dir = Path(r'D:\backup\BaoBao\PythonProgram\miaosuan\data\cache')

h1_full = pd.read_parquet(cache_dir / 'XAUUSD_H1_Dukascopy_full.parquet')
h1_orig = pd.read_parquet(cache_dir / 'XAUUSD_H1.parquet')

h1_full = h1_full.sort_values('time').reset_index(drop=True)
h1_orig = h1_orig.sort_values('time').reset_index(drop=True)

full_max_ts = int(h1_full['time'].max())
print(f"BASIC_LATEST: {datetime.fromtimestamp(full_max_ts, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')}")

patch = h1_orig[h1_orig['time'] > full_max_ts].copy()
print(f"PATCH_ROWS: {len(patch)}")
if len(patch) > 0:
    print(f"PATCH_RANGE: {datetime.fromtimestamp(int(patch['time'].min()), tz=timezone.utc).strftime('%Y-%m-%d %H:%M')} to {datetime.fromtimestamp(int(patch['time'].max()), tz=timezone.utc).strftime('%Y-%m-%d %H:%M')}")

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

print(f"OUTPUT_FILE: {out_path}")
print(f"TOTAL_ROWS: {len(merged)}")
print(f"TIME_RANGE: {datetime.fromtimestamp(int(merged['time'].min()), tz=timezone.utc).strftime('%Y-%m-%d %H:%M')} to {datetime.fromtimestamp(int(merged['time'].max()), tz=timezone.utc).strftime('%Y-%m-%d %H:%M')}")
print(f"UNEXPECTED_GAPS: {len(unexpected)}")
print(f"DIFF_VS_H1_ORIG: {len(merged) - len(h1_orig)}")
print(f"DIFF_VS_H1_FULL: {len(merged) - len(h1_full)}")