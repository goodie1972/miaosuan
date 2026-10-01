import sys, os
sys.path.insert(0, 'src')
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

CACHE_DIR = Path(r'D:\backup\BaoBao\PythonProgram\miaosuan\data\cache')
SYMBOL = 'XAUUSD'
OUT_FILE = Path('build_h4_result.txt')

def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    df = df.copy()
    df['dt'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df = df.set_index('dt').sort_index()
    agg = df.resample(rule).agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
        'volume': 'sum', 'tick_volume': 'sum',
    })
    agg = agg.dropna(subset=['open'])
    agg = agg.reset_index()
    agg['time'] = (agg['dt'].astype('int64') // 1_000_000_000).astype('int64')
    agg = agg.drop(columns=['dt'])
    cols = ['time', 'open', 'high', 'low', 'close', 'volume', 'tick_volume']
    return agg[cols].reset_index(drop=True)

results = []

# Step 1: H4 from H1
h1 = pd.read_parquet(CACHE_DIR / f'{SYMBOL}_H1_Dukascopy_full.parquet')
results.append(f'读取 H1: {len(h1)} 根')

h4 = resample_ohlcv(h1, '4h')
OUT_H4 = CACHE_DIR / f'{SYMBOL}_H4_ultimate.parquet'
tmp = OUT_H4.with_suffix('.tmp')
h4.to_parquet(tmp, index=False)
tmp.replace(OUT_H4)
results.append(f'H4: {len(h4)} 根, 输出到 {OUT_H4}')
results.append(f'  范围: {datetime.fromtimestamp(int(h4.time.min()), tz=timezone.utc)} to {datetime.fromtimestamp(int(h4.time.max()), tz=timezone.utc)}')

# Step 2: Fetch M5 from Dukascopy (long task)
from miaosuan.data.fetchers.dukascopy import DukascopyFetcher
import time

results.append('')
results.append('开始拉取 M5...')

fetcher = DukascopyFetcher(proxy='http://127.0.0.1:10808', timeout=60, window_days=365*20)
t0 = time.time()

raw = fetcher._fetch_raw(SYMBOL, 300)  # 300s = M5

elapsed = time.time() - t0
results.append(f'M5 拉取: {len(raw)} 根, 耗时 {elapsed:.1f}s ({elapsed/60:.1f}min)')

if raw:
    df = pd.DataFrame(raw)
    df = fetcher._finalize(df)
    
    OUT_M5 = CACHE_DIR / f'{SYMBOL}_M5_ultimate.parquet'
    tmp = OUT_M5.with_suffix('.tmp')
    df.to_parquet(tmp, index=False)
    tmp.replace(OUT_M5)
    results.append(f'M5: {len(df)} 根, 输出到 {OUT_M5}')
    results.append(f'  范围: {datetime.fromtimestamp(int(df.time.min()), tz=timezone.utc)} to {datetime.fromtimestamp(int(df.time.max()), tz=timezone.utc)}')
    
    # Step 3: M15 and M30 from M5
    m15 = resample_ohlcv(df, '15min')
    OUT_M15 = CACHE_DIR / f'{SYMBOL}_M15_ultimate.parquet'
    tmp = OUT_M15.with_suffix('.tmp')
    m15.to_parquet(tmp, index=False)
    tmp.replace(OUT_M15)
    results.append(f'M15: {len(m15)} 根')
    results.append(f'  范围: {datetime.fromtimestamp(int(m15.time.min()), tz=timezone.utc)} to {datetime.fromtimestamp(int(m15.time.max()), tz=timezone.utc)}')
    
    m30 = resample_ohlcv(df, '30min')
    OUT_M30 = CACHE_DIR / f'{SYMBOL}_M30_ultimate.parquet'
    tmp = OUT_M30.with_suffix('.tmp')
    m30.to_parquet(tmp, index=False)
    tmp.replace(OUT_M30)
    results.append(f'M30: {len(m30)} 根')
    results.append(f'  范围: {datetime.fromtimestamp(int(m30.time.min()), tz=timezone.utc)} to {datetime.fromtimestamp(int(m30.time.max()), tz=timezone.utc)}')
    
    # Step 4: Append TV latest
    results.append('')
    results.append('用 TV 数据补齐最新...')
    for tf, network_file, ultimate_file in [
        ('H4', f'{SYMBOL}_H4_network.parquet', f'{SYMBOL}_H4_ultimate.parquet'),
        ('M30', f'{SYMBOL}_M30_network.parquet', f'{SYMBOL}_M30_ultimate.parquet'),
        ('M15', f'{SYMBOL}_M15_network.parquet', f'{SYMBOL}_M15_ultimate.parquet'),
    ]:
        net_path = CACHE_DIR / network_file
        ult_path = CACHE_DIR / ultimate_file
        if not net_path.exists() or not ult_path.exists():
            results.append(f'{tf}: 跳过 (文件不存在)')
            continue
        net = pd.read_parquet(net_path)
        ult = pd.read_parquet(ult_path)
        ult_max_ts = int(ult['time'].max())
        patch = net[net['time'] > ult_max_ts]
        if len(patch) == 0:
            results.append(f'{tf}: 无需补齐')
            continue
        merged = pd.concat([ult, patch], ignore_index=True)
        merged = merged.sort_values('time').reset_index(drop=True)
        merged = merged[~merged['time'].duplicated(keep='first')].reset_index(drop=True)
        cols = ['time', 'open', 'high', 'low', 'close', 'volume', 'tick_volume']
        merged = merged[cols]
        tmp = ult_path.with_suffix('.tmp')
        merged.to_parquet(tmp, index=False)
        tmp.replace(ult_path)
        results.append(f'{tf}: 补齐 {len(patch)} 根, 总 {len(merged)} 根')
        results.append(f'  最新: {datetime.fromtimestamp(int(merged.time.max()), tz=timezone.utc)}')

# Final summary
results.append('')
results.append('=== 最终统计 ===')
for name, path in [('H4', OUT_H4), ('M5', OUT_M5), ('M15', OUT_M15), ('M30', OUT_M30)]:
    if path.exists():
        df = pd.read_parquet(path)
        min_dt = datetime.fromtimestamp(int(df.time.min()), tz=timezone.utc).strftime('%Y-%m-%d')
        max_dt = datetime.fromtimestamp(int(df.time.max()), tz=timezone.utc).strftime('%Y-%m-%d')
        results.append(f'{name}: {len(df):>8} 根, {min_dt} to {max_dt}')

with open(OUT_FILE, 'w', encoding='utf-8') as f:
    f.write('\n'.join(results))

print('DONE')