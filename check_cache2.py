import sys, os
sys.path.insert(0, 'src')
import pandas as pd
from pathlib import Path

cache_dir = Path(r'D:\backup\BaoBao\PythonProgram\miaosuan\data\cache')
files = list(cache_dir.glob('*XAUUSD*.parquet'))

out_lines = []
out_lines.append(f"{'文件名':<45} {'行数':>8} {'最早时间':>22} {'最新时间':>22} {'大小(MB)':>8}")
out_lines.append("-" * 110)

for f in sorted(files, key=lambda x: x.stat().st_size, reverse=True):
    try:
        df = pd.read_parquet(f)
        if len(df) == 0:
            out_lines.append(f"{f.name:<45} {'空':>8}")
            continue
        min_ts = int(df['time'].min())
        max_ts = int(df['time'].max())
        from datetime import datetime, timezone
        min_dt = datetime.fromtimestamp(min_ts, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')
        max_dt = datetime.fromtimestamp(max_ts, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')
        size_mb = f.stat().st_size / 1e6
        out_lines.append(f"{f.name:<45} {len(df):>8} {min_dt:>22} {max_dt:>22} {size_mb:>8.2f}")
    except Exception as e:
        out_lines.append(f"{f.name:<45} ERROR: {e}")

with open('check_cache_out.txt', 'w', encoding='utf-8') as out:
    out.write('\n'.join(out_lines))

print('Done')