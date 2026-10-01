import sys, time
sys.path.insert(0, 'src')
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

# 日志
LOG_FILE = Path('D:/backup/BaoBao/PythonProgram/miaosuan/duk_fetch_log.txt')
def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    line = f'{ts} {msg}'
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

CACHE_DIR = Path(r'D:\backup\BaoBao\PythonProgram\miaosuan\data\cache')
SYMBOL = 'XAUUSD'
PROXY = 'http://127.0.0.1:10808'

# 清空日志
open(LOG_FILE, 'w').close()

log('=== Dukascopy 全周期数据拉取开始 ===')
log(f'开始时间: {datetime.now()}')

# Step 1: 拉取 M5（最细粒度，覆盖 M15/M30 的所有数据）
from miaosuan.data.fetchers.dukascopy import DukascopyFetcher, _TF_SECONDS, _DUK_TIMEFRAME_MAP

log(f'拉取 {SYMBOL} M5 全量 (20年)...')
fetcher = DukascopyFetcher(proxy=PROXY, timeout=60, window_days=365*20)

t0 = time.time()
df_m5 = fetcher.fetch_full(SYMBOL, 'M5')
elapsed = time.time() - t0
log(f'M5 完成: {len(df_m5)} 根, 耗时 {elapsed:.0f}s ({elapsed/60:.1f}min)')

if len(df_m5) > 0:
    # 写入 M5
    OUT_M5 = CACHE_DIR / f'{SYMBOL}_M5_ultimate.parquet'
    df_m5.to_parquet(OUT_M5, index=False)
    min_dt = datetime.fromtimestamp(int(df_m5['time'].min()), tz=timezone.utc)
    max_dt = datetime.fromtimestamp(int(df_m5['time'].max()), tz=timezone.utc)
    log(f'M5: {len(df_m5)} 根, {min_dt} to {max_dt}, 大小 {OUT_M5.stat().st_size/1e6:.2f}MB')

    # Step 2: 从 M5 重采样 M15 和 M30
    def resample(df, rule):
        d = df.copy()
        d['dt'] = pd.to_datetime(d['time'], unit='s', utc=True)
        d = d.set_index('dt').sort_index()
        agg = d.resample(rule).agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum','tick_volume':'sum'})
        agg = agg.dropna(subset=['open']).reset_index()
        agg['time'] = (agg['dt'].astype('int64') // 1_000_000_000).astype('int64')
        return agg[['time','open','high','low','close','volume','tick_volume']].reset_index(drop=True)

    df_m15 = resample(df_m5, '15min')
    OUT_M15 = CACHE_DIR / f'{SYMBOL}_M15_ultimate.parquet'
    df_m15.to_parquet(OUT_M15, index=False)
    log(f'M15: {len(df_m15)} 根, {datetime.fromtimestamp(int(df_m15.time.min()), tz=timezone.utc)} to {datetime.fromtimestamp(int(df_m15.time.max()), tz=timezone.utc)}')

    df_m30 = resample(df_m5, '30min')
    OUT_M30 = CACHE_DIR / f'{SYMBOL}_M30_ultimate.parquet'
    df_m30.to_parquet(OUT_M30, index=False)
    log(f'M30: {len(df_m30)} 根, {datetime.fromtimestamp(int(df_m30.time.min()), tz=timezone.utc)} to {datetime.fromtimestamp(int(df_m30.time.max()), tz=timezone.utc)}')

    # Step 3: 用 TV 补齐最新（M5/M15/M30/H4）
    log('用 TV 数据补齐最新...')
    for tf_name, network_file in [('M30', 'XAUUSD_M30_network.parquet'), ('M15', 'XAUUSD_M15_network.parquet')]:
        ult_path = CACHE_DIR / f'{SYMBOL}_{tf_name}_ultimate.parquet'
        net_path = CACHE_DIR / network_file
        if not net_path.exists():
            log(f'{tf_name}: TV 文件不存在，跳过')
            continue
        net = pd.read_parquet(net_path)
        ult = pd.read_parquet(ult_path)
        ult_max = int(ult['time'].max())
        patch = net[net['time'] > ult_max]
        if len(patch) == 0:
            log(f'{tf_name}: TV 无更新')
            continue
        merged = pd.concat([ult, patch], ignore_index=True)
        merged = merged.sort_values('time').drop_duplicates('time', keep='first').reset_index(drop=True)
        merged.to_parquet(ult_path, index=False)
        log(f'{tf_name}: 补齐 {len(patch)} 根, 总 {len(merged)} 根, 最新 {datetime.fromtimestamp(int(merged.time.max()), tz=timezone.utc)}')

    # H4 也已生成（从 build_h4.py），也补齐一下
    net_path = CACHE_DIR / 'XAUUSD_H4_network.parquet'
    ult_path = CACHE_DIR / f'{SYMBOL}_H4_ultimate.parquet'
    if net_path.exists() and ult_path.exists():
        net = pd.read_parquet(net_path)
        ult = pd.read_parquet(ult_path)
        ult_max = int(ult['time'].max())
        patch = net[net['time'] > ult_max]
        if len(patch) > 0:
            merged = pd.concat([ult, patch], ignore_index=True)
            merged = merged.sort_values('time').drop_duplicates('time', keep='first').reset_index(drop=True)
            merged.to_parquet(ult_path, index=False)
            log(f'H4: 补齐 {len(patch)} 根, 总 {len(merged)} 根')
        else:
            log(f'H4: TV 无更新')

log('=== 全部完成 ===')
log(f'结束时间: {datetime.now()}')
log(f'总耗时: {time.time()-t0:.0f}s ({(time.time()-t0)/3600:.1f}h)')