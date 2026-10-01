import sys, time, json, urllib.request, calendar
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, 'src')
from miaosuan.data.fetchers.dukascopy import DukascopyFetcher

LOG_FILE = Path('D:/backup/BaoBao/PythonProgram/miaosuan/duk_fetch_log.txt')
CACHE_DIR = Path(r'D:\backup\BaoBao\PythonProgram\miaosuan\data\cache')
SYMBOL = 'XAUUSD'
PROXY = 'http://127.0.0.1:10808'

def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    line = f'{ts} {msg}'
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

open(LOG_FILE, 'w').close()
log('=== Dukascopy 全周期直接拉取开始 ===')
log(f'开始时间: {datetime.now()}')

# 创建 fetcher 实例（复用 _parse_response 和 _finalize）
fetcher = DukascopyFetcher(proxy=PROXY, timeout=60, window_days=365*20)

# ── 1. 直接拉取 H4（从 /hour/ endpoint）──
log('--- 拉取 H4 (直接从 Dukascopy /hour/ endpoint) ---')
t0 = time.time()
try:
    df_h4 = fetcher.fetch_full(SYMBOL, 'H4')
    elapsed = time.time() - t0
    OUT_H4 = CACHE_DIR / f'{SYMBOL}_H4_Dukascopy.parquet'
    df_h4.to_parquet(OUT_H4, index=False)
    log(f'H4: {len(df_h4)} 根, {elapsed:.0f}s ({elapsed/60:.1f}min)')
    log(f'  范围: {datetime.fromtimestamp(int(df_h4["time"].min()), tz=timezone.utc)} to {datetime.fromtimestamp(int(df_h4["time"].max()), tz=timezone.utc)}')
    log(f'  输出: {OUT_H4}')
except Exception as e:
    log(f'H4 ERROR: {e}')

# ── 2. 直接拉取 D1（从 /day/ endpoint）──
log('--- 拉取 D1 (直接从 Dukascopy /day/ endpoint) ---')
t0 = time.time()
try:
    df_d1 = fetcher.fetch_full(SYMBOL, 'D1')
    elapsed = time.time() - t0
    OUT_D1 = CACHE_DIR / f'{SYMBOL}_D1_Dukascopy_direct.parquet'
    df_d1.to_parquet(OUT_D1, index=False)
    log(f'D1: {len(df_d1)} 根, {elapsed:.0f}s ({elapsed/60:.1f}min)')
    log(f'  范围: {datetime.fromtimestamp(int(df_d1["time"].min()), tz=timezone.utc)} to {datetime.fromtimestamp(int(df_d1["time"].max()), tz=timezone.utc)}')
    log(f'  输出: {OUT_D1}')
except Exception as e:
    log(f'D1 ERROR: {e}')

# ── 3. 直接拉取 M5（从 /minute/ endpoint，最耗时）──
log('--- 拉取 M5 (直接从 Dukascopy /minute/ endpoint) ---')
log('  这是最耗时的部分：约 5000 个交易日，每分钟级数据量大')
t0 = time.time()

handler = urllib.request.ProxyHandler({'http': PROXY, 'https': PROXY})
opener = urllib.request.build_opener(handler)

all_rows_m5 = []
day_count = 0
day_fail = 0
last_log = 0
t_m5_start = time.time()

for year in range(2006, 2027):
    for month in range(1, 13):
        if year == 2026 and month > 9:
            break
        days_in_month = calendar.monthrange(year, month)[1]
        for day in range(1, days_in_month + 1):
            dt = datetime(year, month, day, tzinfo=timezone.utc)
            if dt.weekday() >= 5:
                continue

            url = f'https://jetta.dukascopy.com/v1/candles/minute/XAU-USD/BID/{year}/{month}/{day}'
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'MiaoSuan/1.0'})
                with opener.open(req, timeout=30) as resp:
                    body = resp.read()
                    data = json.loads(body.decode('utf-8'))
                rows = fetcher._parse_response(data, 300)
                all_rows_m5.extend(rows)
                day_count += 1
            except Exception as e:
                day_fail += 1

            if day_count - last_log >= 200:
                elapsed = time.time() - t_m5_start
                rate = day_count / elapsed if elapsed > 0 else 0
                remaining = (5000 - day_count) / rate if rate > 0 else 0
                log(f'  M5 进度: {day_count} 天成功, {day_fail} 失败, {elapsed:.0f}s, ~{rate:.1f}天/s, 预计剩余 {remaining/60:.0f}min')
                last_log = day_count

elapsed_m5 = time.time() - t_m5_start
log(f'M5 拉取完成: {len(all_rows_m5)} 根, {day_count} 天成功, {day_fail} 天失败, 耗时 {elapsed_m5:.0f}s ({elapsed_m5/60:.1f}min)')

if all_rows_m5:
    df_m5 = pd.DataFrame(all_rows_m5)
    df_m5 = fetcher._finalize(df_m5)
    
    OUT_M5 = CACHE_DIR / f'{SYMBOL}_M5_Dukascopy.parquet'
    df_m5.to_parquet(OUT_M5, index=False)
    log(f'M5: {len(df_m5)} 根, 输出到 {OUT_M5}')
    log(f'  范围: {datetime.fromtimestamp(int(df_m5["time"].min()), tz=timezone.utc)} to {datetime.fromtimestamp(int(df_m5["time"].max()), tz=timezone.utc)}')

    # ── 4. M15 和 M30 直接从 Dukascopy 拉取（也是 /minute/ endpoint）──
    # 但实际上 M15/M30 的 API 也是 minute 级别的，只是过滤条件不同
    # 更高效的方式：从已有的 M5 数据重采样（秒级）
    log('--- 从 M5 重采样 M15/M30 ---')
    
    def resample(df, rule):
        d = df.copy()
        d['dt'] = pd.to_datetime(d['time'], unit='s', utc=True)
        d = d.set_index('dt').sort_index()
        agg = d.resample(rule).agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum','tick_volume':'sum'})
        agg = agg.dropna(subset=['open']).reset_index()
        agg['time'] = (agg['dt'].astype('int64') // 1_000_000_000).astype('int64')
        return agg[['time','open','high','low','close','volume','tick_volume']].reset_index(drop=True)
    
    df_m15 = resample(df_m5, '15min')
    OUT_M15 = CACHE_DIR / f'{SYMBOL}_M15_Dukascopy.parquet'
    df_m15.to_parquet(OUT_M15, index=False)
    log(f'M15: {len(df_m15)} 根, 输出到 {OUT_M15}')
    
    df_m30 = resample(df_m5, '30min')
    OUT_M30 = CACHE_DIR / f'{SYMBOL}_M30_Dukascopy.parquet'
    df_m30.to_parquet(OUT_M30, index=False)
    log(f'M30: {len(df_m30)} 根, 输出到 {OUT_M30}')

# ── 5. 用 TV 补齐最新数据 ──
log('--- 用 TV 数据补齐最新 ---')
for tf_name, duk_file, network_file in [
    ('H4', f'{SYMBOL}_H4_Dukascopy.parquet', f'{SYMBOL}_H4_network.parquet'),
    ('M5', f'{SYMBOL}_M5_Dukascopy.parquet', None),
    ('M15', f'{SYMBOL}_M15_Dukascopy.parquet', f'{SYMBOL}_M15_network.parquet'),
    ('M30', f'{SYMBOL}_M30_Dukascopy.parquet', f'{SYMBOL}_M30_network.parquet'),
]:
    duk_path = CACHE_DIR / duk_file
    if not duk_path.exists():
        log(f'{tf_name}: Dukascopy 文件不存在，跳过')
        continue
    if network_file is None:
        log(f'{tf_name}: 无 TV 文件，跳过补齐')
        continue
    net_path = CACHE_DIR / network_file
    if not net_path.exists():
        log(f'{tf_name}: TV 文件不存在，跳过')
        continue
    net = pd.read_parquet(net_path)
    duk = pd.read_parquet(duk_path)
    duk_max = int(duk['time'].max())
    patch = net[net['time'] > duk_max]
    if len(patch) == 0:
        log(f'{tf_name}: TV 无更新')
        continue
    merged = pd.concat([duk, patch], ignore_index=True)
    merged = merged.sort_values('time').drop_duplicates('time', keep='first').reset_index(drop=True)
    merged.to_parquet(duk_path, index=False)
    log(f'{tf_name}: 补齐 {len(patch)} 根, 总 {len(merged)} 根, 最新 {datetime.fromtimestamp(int(merged["time"].max()), tz=timezone.utc)}')

# ── 最终统计 ──
log('\n=== 最终统计 ===')
for name, path in [('H4', CACHE_DIR / f'{SYMBOL}_H4_Dukascopy.parquet'),
                   ('D1', CACHE_DIR / f'{SYMBOL}_D1_Dukascopy_direct.parquet'),
                   ('M5', CACHE_DIR / f'{SYMBOL}_M5_Dukascopy.parquet'),
                   ('M15', CACHE_DIR / f'{SYMBOL}_M15_Dukascopy.parquet'),
                   ('M30', CACHE_DIR / f'{SYMBOL}_M30_Dukascopy.parquet')]:
    if path.exists():
        df = pd.read_parquet(path)
        min_dt = datetime.fromtimestamp(int(df['time'].min()), tz=timezone.utc).strftime('%Y-%m-%d')
        max_dt = datetime.fromtimestamp(int(df['time'].max()), tz=timezone.utc).strftime('%Y-%m-%d')
        size = path.stat().st_size / 1e6
        log(f'{name}: {len(df):>8} 根, {min_dt} to {max_dt}, {size:.2f}MB')
    else:
        log(f'{name}: 不存在')

log(f'\n=== 全部完成 ===')
log(f'结束时间: {datetime.now()}')
log(f'总耗时: {(time.time() - t_m5_start)/60:.0f}min')