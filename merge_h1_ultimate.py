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
print(f"基础最新时间: {datetime.fromtimestamp(full_max_ts, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')}")

patch = h1_orig[h1_orig['time'] > full_max_ts].copy()
print(f"补丁行数: {len(patch)}")
if len(patch) > 0:
    print(f"补丁范围: {datetime.fromtimestamp(int(patch['time'].min()), tz=timezone.utc).strftime('%Y-%m-%d %H:%M')} → "
          f"{datetime.fromtimestamp(int(patch['time'].max()), tz=timezone.utc).strftime('%Y-%m-%d %H:%M')}")

merged = pd.concat([h1_full, patch], ignore_index=True)
merged = merged.sort_values('time').reset_index(drop=True)
merged = merged[~merged['time'].duplicated(keep='first')].reset_index(drop=True)

cols = ['time', 'open', 'high', 'low', 'close', 'volume', 'tick_volume']
merged = merged[cols]

out_path = cache_dir / 'XAUUSD_H1_ultimate.parquet'
tmp = out_path.with_suffix('.tmp')
merged.to_parquet(tmp, index=False)
tmp.replace(out_path)

print(f"\n=== 合并完成 ===")
print(f"输出文件: {out_path}")
print(f"总行数: {len(merged)}")
print(f"时间范围: {datetime.fromtimestamp(int(merged['time'].min()), tz=timezone.utc).strftime('%Y-%m-%d %H:%M')} → "
      f"{datetime.fromtimestamp(int(merged['time'].max()), tz=timezone.utc).strftime('%Y-%m-%d %H:%M')}")

# 验证连续性（H1 允许周末间隙）
diffs = merged['time'].diff().dropna().astype(int)
# 正常间隙：1h=3600, 2h=7200(周末隔夜), 3h=10800(周五到周一), 4h=14400(节假日)
expected = {3600, 7200, 10800, 14400}
unexpected = diffs[~diffs.isin(expected)]
print(f"异常间隙: {len(unexpected)} 个")
if len(unexpected) > 0:
    print(f"  样例: {unexpected.head(10).tolist()}")

print(f"\n与 XAUUSD_H1.parquet 差异: {len(merged) - len(h1_orig)} 行")
print(f"与 XAUUSD_H1_Dukascopy_full.parquet 差异: {len(merged) - len(h1_full)} 行")