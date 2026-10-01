import sys, pandas as pd, numpy as np
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, 'src')
cache_dir = Path(r'D:\backup\BaoBao\PythonProgram\miaosuan\data\cache')
out_lines = []

d1_full = pd.read_parquet(cache_dir / 'XAUUSD_D1_Dukascopy_full.parquet')
d1_orig = pd.read_parquet(cache_dir / 'XAUUSD_D1.parquet')

d1_full = d1_full.sort_values('time').reset_index(drop=True)
d1_orig = d1_orig.sort_values('time').reset_index(drop=True)

full_max_ts = int(d1_full['time'].max())
out_lines.append(f"基础最新时间: {datetime.fromtimestamp(full_max_ts, tz=timezone.utc).strftime('%Y-%m-%d')}")

patch = d1_orig[d1_orig['time'] > full_max_ts].copy()
out_lines.append(f"补丁行数: {len(patch)}")
if len(patch) > 0:
    out_lines.append(f"补丁范围: {datetime.fromtimestamp(int(patch['time'].min()), tz=timezone.utc).strftime('%Y-%m-%d')} → "
          f"{datetime.fromtimestamp(int(patch['time'].max()), tz=timezone.utc).strftime('%Y-%m-%d')}")

merged = pd.concat([d1_full, patch], ignore_index=True)
merged = merged.sort_values('time').reset_index(drop=True)
merged = merged[~merged['time'].duplicated(keep='first')].reset_index(drop=True)

cols = ['time', 'open', 'high', 'low', 'close', 'volume', 'tick_volume']
merged = merged[cols]

out_path = cache_dir / 'XAUUSD_D1_ultimate.parquet'
tmp = out_path.with_suffix('.tmp')
merged.to_parquet(tmp, index=False)
tmp.replace(out_path)

out_lines.append(f"\n=== 合并完成 ===")
out_lines.append(f"输出文件: {out_path}")
out_lines.append(f"总行数: {len(merged)}")
out_lines.append(f"时间范围: {datetime.fromtimestamp(int(merged['time'].min()), tz=timezone.utc).strftime('%Y-%m-%d')} → "
      f"{datetime.fromtimestamp(int(merged['time'].max()), tz=timezone.utc).strftime('%Y-%m-%d')}")

diffs = merged['time'].diff().dropna().astype(int)
unexpected = diffs[(diffs != 86400) & (diffs != 172800) & (diffs != 259200)]
out_lines.append(f"异常间隙: {len(unexpected)} 个")
if len(unexpected) > 0:
    out_lines.append(f"  样例: {unexpected.head(10).tolist()}")

out_lines.append(f"\n与 XAUUSD_D1.parquet 差异: {len(merged) - len(d1_orig)} 行")
out_lines.append(f"与 XAUUSD_D1_Dukascopy_full.parquet 差异: {len(merged) - len(d1_full)} 行")

with open('merge_ultimate_out.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(out_lines))

print('Done, check merge_ultimate_out.txt')