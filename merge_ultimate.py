import sys, pandas as pd, numpy as np
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, 'src')
cache_dir = Path(r'D:\backup\BaoBao\PythonProgram\miaosuan\data\cache')

# 读取两个源
d1_full = pd.read_parquet(cache_dir / 'XAUUSD_D1_Dukascopy_full.parquet')  # 基础：2006-10 → 2026-08-31
d1_orig = pd.read_parquet(cache_dir / 'XAUUSD_D1.parquet')                   # 补丁：含 2026-09

# 基础按 time 排序
d1_full = d1_full.sort_values('time').reset_index(drop=True)
d1_orig = d1_orig.sort_values('time').reset_index(drop=True)

# 找出基础的最新时间
full_max_ts = int(d1_full['time'].max())
print(f"基础最新时间: {datetime.fromtimestamp(full_max_ts, tz=timezone.utc).strftime('%Y-%m-%d')}")

# 从原数据中取 > 基础最新时间的行
patch = d1_orig[d1_orig['time'] > full_max_ts].copy()
print(f"补丁行数: {len(patch)}")
if len(patch) > 0:
    print(f"补丁范围: {datetime.fromtimestamp(int(patch['time'].min()), tz=timezone.utc).strftime('%Y-%m-%d')} → "
          f"{datetime.fromtimestamp(int(patch['time'].max()), tz=timezone.utc).strftime('%Y-%m-%d')}")

# 合并：基础 + 补丁，去重保留基础（source 更权威）
merged = pd.concat([d1_full, patch], ignore_index=True)
merged = merged.sort_values('time').reset_index(drop=True)
merged = merged[~merged['time'].duplicated(keep='first')].reset_index(drop=True)  # keep='first' 保留基础

# 标准列顺序
cols = ['time', 'open', 'high', 'low', 'close', 'volume', 'tick_volume']
merged = merged[cols]

# 写入
out_path = cache_dir / 'XAUUSD_D1_ultimate.parquet'
tmp = out_path.with_suffix('.tmp')
merged.to_parquet(tmp, index=False)
tmp.replace(out_path)

print(f"\n=== 合并完成 ===")
print(f"输出文件: {out_path}")
print(f"总行数: {len(merged)}")
print(f"时间范围: {datetime.fromtimestamp(int(merged['time'].min()), tz=timezone.utc).strftime('%Y-%m-%d')} → "
      f"{datetime.fromtimestamp(int(merged['time'].max()), tz=timezone.utc).strftime('%Y-%m-%d')}")

# 验证连续性
diffs = merged['time'].diff().dropna().astype(int)
unexpected = diffs[(diffs != 86400) & (diffs != 172800) & (diffs != 259200)]
print(f"异常间隙: {len(unexpected)} 个")
if len(unexpected) > 0:
    print(f"  样例: {unexpected.head(10).tolist()}")

# 与原文件对比
print(f"\n与 XAUUSD_D1.parquet 差异: {len(merged) - len(d1_orig)} 行")
print(f"与 XAUUSD_D1_Dukascopy_full.parquet 差异: {len(merged) - len(d1_full)} 行")