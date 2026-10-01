import sys, time, json, urllib.request
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, 'src')

LOG_FILE = Path('D:/backup/BaoBao/PythonProgram/miaosuan/duk_m5_log.txt')
CACHE_DIR = Path(r'D:\backup\BaoBao\PythonProgram\miaosuan\data\cache')
SYMBOL = 'XAUUSD'
PROXY = 'http://127.0.0.1:10808'

def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    line = f'{ts} {msg}'
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

open(LOG_FILE, 'w').close()
t_start = time.time()
log('=== Dukascopy M5 全量拉取开始 ===')
log(f'开始时间: {datetime.now()}')

# 使用 DukascopyFetcher 的代理 opener
handler = urllib.request.ProxyHandler({'http': PROXY, 'https': PROXY})
opener = urllib.request.build_opener(handler)

# 从 dukascopy.py 复用解析逻辑
from miaosuan.data.fetchers.dukascopy import DukascopyFetcher
_fetcher = DukascopyFetcher(proxy=PROXY, timeout=60)

# 手动拉取，带进度日志
all_rows = []
duk_symbol = 'XAU-USD'
t0 = time.time()
day_count = 0
day_fail = 0
last_log = 0

for year in range(2006, 2027):
    import calendar
    for month in range(1, 13):
        if year == 2026 and month > 9:
            break
        days_in_month = calendar.monthrange(year, month)[1]
        for day in range(1, days_in_month + 1):
            dt = datetime(year, month, day, tzinfo=timezone.utc)
            if dt.weekday() >= 5:
                continue

            url = f'https://jetta.dukascopy.com/v1/candles/minute/{duk_symbol}/BID/{year}/{month}/{day}'
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'MiaoSuan/1.0'})
                with opener.open(req, timeout=30) as resp:
                    body = resp.read()
                    data = json.loads(body.decode('utf-8'))
                rows = _parse_response(data, 300)
                all_rows.extend(rows)
                day_count += 1
            except Exception as e:
                day_fail += 1

            if day_count - last_log >= 200:
                elapsed = time.time() - t0
                rate = day_count / elapsed if elapsed > 0 else 0
                remaining = (5000 - day_count) / rate if rate > 0 else 0
                log(f'  进度: {day_count} 天, 失败 {day_fail}, {elapsed:.0f}s, ~{rate:.1f}天/s, 预计剩余 {remaining/60:.0f}min')
                last_log = day_count

elapsed = time.time() - t0
log(f'M5 拉取完成: {len(all_rows)} 根, {day_count} 天成功, {day_fail} 天失败, 耗时 {elapsed:.0f}s ({elapsed/60:.1f}min)')

# 构建 DataFrame
if all_rows:
    df = pd.DataFrame(all_rows)
    # finalize
    df['time'] = df['time'].astype('int64')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype('float64')
    if 'tick_volume' not in df.columns:
        df['tick_volume'] = df['volume']
    df = df.sort_values('time').drop_duplicates('time', keep='last').reset_index(drop=True)
    df = df[['time','open','high','low','close','volume','tick_volume']]
    
    OUT_M5 = CACHE_DIR / f'{SYMBOL}_M5_ultimate.parquet'
    df.to_parquet(OUT_M5, index=False)
    min_dt = datetime.fromtimestamp(int(df['time'].min()), tz=timezone.utc)
    max_dt = datetime.fromtimestamp(int(df['time'].max()), tz=timezone.utc)
    log(f'M5: {len(df)} 根, {min_dt} to {max_dt}, {OUT_M5.stat().st_size/1e6:.2f}MB')

    # M15 and M30 from M5
    def resample(df, rule):
        d = df.copy()
        d['dt'] = pd.to_datetime(d['time'], unit='s', utc=True)
        d = d.set_index('dt').sort_index()
        agg = d.resample(rule).agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum','tick_volume':'sum'})
        agg = agg.dropna(subset=['open']).reset_index()
        agg['time'] = (agg['dt'].astype('int64') // 1_000_000_000).astype('int64')
        return agg[['time','open','high','low','close','volume','tick_volume']].reset_index(drop=True)

    df_m15 = resample(df, '15min')
    OUT_M15 = CACHE_DIR / f'{SYMBOL}_M15_ultimate.parquet'
    df_m15.to_parquet(OUT_M15, index=False)
    log(f'M15: {len(df_m15)} 根')

    df_m30 = resample(df, '30min')
    OUT_M30 = CACHE_DIR / f'{SYMBOL}_M30_ultimate.parquet'
    df_m30.to_parquet(OUT_M30, index=False)
    log(f'M30: {len(df_m30)} 根')

    # TV 补齐
    for tf_name, network_file in [('M30', 'XAUUSD_M30_network.parquet'), ('M15', 'XAUUSD_M15_network.parquet')]:
        ult_path = CACHE_DIR / f'{SYMBOL}_{tf_name}_ultimate.parquet'
        net_path = CACHE_DIR / network_file
        if not net_path.exists():
            continue
        net = pd.read_parquet(net_path)
        ult = pd.read_parquet(ult_path)
        ult_max = int(ult['time'].max())
        patch = net[net['time'] > ult_max]
        if len(patch) > 0:
            merged = pd.concat([ult, patch], ignore_index=True)
            merged = merged.sort_values('time').drop_duplicates('time', keep='first').reset_index(drop=True)
            merged.to_parquet(ult_path, index=False)
            log(f'{tf_name}: 补齐 {len(patch)} 根, 总 {len(merged)} 根')
    log(f'=== 全部完成 ===')
else:
    log('ERROR: 无数据')