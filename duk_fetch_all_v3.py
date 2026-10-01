"""
Dukascopy 全周期直接拉取 v3
- 顺序：M30 → M15 → M5（从小到大，先快速出结果）
- 断点续传：checkpoint JSON 记录每个周期的当前日期位置
- 严格超时：socket.setdefaulttimeout + 线程级硬超时
- 增量保存：每 500 天保存一次 parquet，崩溃不丢数据
"""
import sys, os, time, json, urllib.request, calendar, socket, threading, queue
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

sys.path.insert(0, 'src')
from miaosuan.data.fetchers.dukascopy import DukascopyFetcher

# ── 配置 ──
LOG_FILE = Path('D:/backup/BaoBao/PythonProgram/miaosuan/duk_fetch_log.txt')
CHECKPOINT_FILE = Path('D:/backup/BaoBao/PythonProgram/miaosuan/tmp/duk_checkpoint.json')
CACHE_DIR = Path(r'D:\backup\BaoBao\PythonProgram\miaosuan\data\cache')
SYMBOL = 'XAUUSD'
DUK_SYMBOL = 'XAU-USD'
PROXY = 'http://127.0.0.1:10808'
SOCKET_TIMEOUT = 60        # socket 全局超时（Dukascopy API 响应较慢）
REQUEST_TIMEOUT = 90       # 线程级硬超时（含 socket 超时 + 余量）
SAVE_EVERY = 50           # 每 50 天保存一次 parquet
PROGRESS_EVERY = 10       # 每 10 天输出一次进度日志
CP_EVERY = 5              # 每 5 天保存一次 checkpoint（防止崩溃丢失进度）

# 周期配置：(名称, 解析间隔秒数, 预期总量)
TIMEFRAMES = [
    ('M30', 1800, 30000),
    ('M15', 900, 60000),
    ('M5', 300, 180000),
]

# ── 全局 socket 超时（解决 hang 住的核心）──
socket.setdefaulttimeout(SOCKET_TIMEOUT)

# ── 日志 ──
def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    line = f'{ts} {msg}'
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')
    print(line, flush=True)

# ── 断点管理 ──
def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {tf: {'status': 'pending'} for tf, _, _ in TIMEFRAMES}

def save_checkpoint(cp):
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
        json.dump(cp, f, ensure_ascii=False, indent=2)

# ── 带硬超时的请求 ──
handler = urllib.request.ProxyHandler({'http': PROXY, 'https': PROXY})
opener = urllib.request.build_opener(handler)
fetcher = DukascopyFetcher(proxy=PROXY, timeout=60, window_days=365 * 20)

def fetch_day(url, timeout=REQUEST_TIMEOUT):
    """使用 fetcher 内置重试机制的单天请求"""
    from miaosuan.errors import DataError
    try:
        return fetcher._http_get(url)
    except DataError as e:
        err_str = str(e)
        # 超时或 400 错误都跳过
        if 'timeout' in err_str.lower() or 'timed out' in err_str.lower() or '400' in err_str:
            raise DataError(err_str)
        raise RuntimeError(err_str)

# ── 生成所有交易日 ──
# Dukascopy XAUUSD 数据从 2006-10-05 开始，跳过之前无数据月份
DUK_START_YEAR = 2006
DUK_START_MONTH = 10
DUK_START_DAY = 5

def generate_days(start_date=None):
    """生成 2006-10-05 至 2026-09 所有工作日（跳过周末）"""
    days = []
    for year in range(DUK_START_YEAR, 2027):
        for month in range(1, 13):
            if year == 2026 and month > 9:
                break
            # 跳过 2006 年 10 月之前的月份
            if year == DUK_START_YEAR and month < DUK_START_MONTH:
                continue
            if year == DUK_START_YEAR and month == DUK_START_MONTH and 1 >= DUK_START_DAY:
                continue
            days_in_month = calendar.monthrange(year, month)[1]
            for day in range(1, days_in_month + 1):
                # 跳过起始日期之前的天
                if year == DUK_START_YEAR and month == DUK_START_MONTH and day < DUK_START_DAY:
                    continue
                dt = datetime(year, month, day, tzinfo=timezone.utc)
                if dt.weekday() >= 5:
                    continue
                days.append((year, month, day))
    
    if start_date:
        sy, sm, sd = start_date
        # 找到起始位置
        start_dt = datetime(sy, sm, sd, tzinfo=timezone.utc)
        days = [(y, m, d) for y, m, d in days 
                if datetime(y, m, d, tzinfo=timezone.utc) > start_dt]
    return days

# ── 加载已保存的部分数据 ──
def load_partial_parquet(tf_name):
    """加载已有的 parquet 文件（如果有）"""
    path = CACHE_DIR / f'{SYMBOL}_{tf_name}_Dukascopy.parquet'
    if path.exists():
        try:
            df = pd.read_parquet(path)
            return df
        except Exception:
            pass
    return None

def save_partial_parquet(tf_name, rows):
    """增量保存数据到 parquet"""
    path = CACHE_DIR / f'{SYMBOL}_{tf_name}_Dukascopy.parquet'
    if not rows:
        return
    df = pd.DataFrame(rows)
    df = fetcher._finalize(df)
    df.to_parquet(path, index=False)

# ── 主流程 ──
open(LOG_FILE, 'w').close()
log('=== Dukascopy 全周期直接拉取 v3 开始 ===')
log(f'开始时间: {datetime.now()}')
log(f'顺序: M30 → M15 → M5')
log(f'超时: socket={SOCKET_TIMEOUT}s, request={REQUEST_TIMEOUT}s')

cp = load_checkpoint()
all_days = generate_days()
log(f'总交易日: {len(all_days)} 天')

# ── H4 和 D1 已存在，跳过 ──
log('\n--- H4, D1 已存在，跳过 ---')

# ── M30 → M15 → M5 ──
for tf_name, interval_sec, est_bars in TIMEFRAMES:
    cp_entry = cp.get(tf_name, {'status': 'pending'})
    
    # 已完成则跳过
    if cp_entry.get('status') == 'completed':
        log(f'\n--- {tf_name}: 已完成，跳过 ---')
        continue
    
    log(f'\n--- 拉取 {tf_name} (Dukascopy /minute/ endpoint) ---')
    cp_entry['status'] = 'in_progress'
    save_checkpoint(cp)
    
    # 检查是否有之前保存的部分数据
    start_date = None
    existing_rows = []
    if 'year' in cp_entry and 'month' in cp_entry and 'day' in cp_entry:
        start_date = (cp_entry['year'], cp_entry['month'], cp_entry['day'])
        log(f'  从断点恢复: {start_date[0]}-{start_date[1]:02d}-{start_date[2]:02d}')
    
    # 加载已保存的部分数据
    partial = load_partial_parquet(tf_name)
    if partial is not None:
        existing_rows = partial.to_dict('records')
        log(f'  已加载 {len(existing_rows)} 根已保存数据')
    
    days = generate_days(start_date)
    total_days = len(days)
    
    day_count = cp_entry.get('days_done', 0)
    day_fail = cp_entry.get('days_failed', 0)
    day_ok = cp_entry.get('days_ok', 0)
    last_log_count = day_count
    last_save_count = day_count
    last_cp_count = day_count
    t_start = time.time()
    
    for i, (year, month, day) in enumerate(days):
        url = f'https://jetta.dukascopy.com/v1/candles/minute/{DUK_SYMBOL}/BID/{year}/{month}/{day}'
        try:
            data = fetch_day(url)  # 返回已解析的 JSON
            rows = fetcher._parse_response(data, interval_sec)
            existing_rows.extend(rows)
            day_count += 1
            day_ok += 1
        except Exception as e:
            day_fail += 1
            # 记录失败详情（帮助排查）
            if day_fail <= 30:
                log(f'  {year}-{month:02d}-{day:02d} FAIL: {str(e)[:100]}')
        
        # 更新 checkpoint
        cp_entry['year'] = year
        cp_entry['month'] = month
        cp_entry['day'] = day
        cp_entry['days_done'] = day_count
        cp_entry['days_failed'] = day_fail
        cp_entry['days_ok'] = day_ok
        cp_entry['last_update'] = datetime.now().isoformat()
        cp_entry['current_date'] = f'{year}-{month:02d}-{day:02d}'
        cp_entry['bars_count'] = len(existing_rows)
        
        # 进度日志（每 10 天）
        if day_count - last_log_count >= PROGRESS_EVERY:
            elapsed = time.time() - t_start
            total_processed = day_count + day_fail
            rate = total_processed / elapsed if elapsed > 0 else 0
            remaining_days = total_days - (i + 1)
            remaining_sec = remaining_days / rate if rate > 0 else 0
            log(f'  {tf_name} 进度: {day_ok}成功/{day_fail}失败/{day_count}有效, {day_count}/{total_days}天 ({day_count/total_days*100:.0f}%), '
                f'{elapsed:.0f}s, ~{rate:.1f}天/s, '
                f'范围: {all_days[0][0]}-{all_days[0][1]:02d}-{all_days[0][2]:02d} → {year}-{month:02d}-{day:02d}, '
                f'预计剩余 {remaining_sec/60:.0f}min')
            last_log_count = day_count
        
        # 频繁保存 checkpoint（每 5 天，防止崩溃丢失）
        if (i + 1) % CP_EVERY == 0:
            save_checkpoint(cp)
            last_cp_count = i + 1
        
        # 增量保存 parquet（每 50 天）
        if day_count - last_save_count >= SAVE_EVERY:
            save_partial_parquet(tf_name, existing_rows)
            save_checkpoint(cp)
            last_save_count = day_count
    
    # 保存最终数据
    elapsed = time.time() - t_start
    save_partial_parquet(tf_name, existing_rows)
    cp_entry['status'] = 'completed'
    cp_entry['total_bars'] = len(existing_rows)
    cp_entry['completed_at'] = datetime.now().isoformat()
    save_checkpoint(cp)
    
    # 输出结果
    out_path = CACHE_DIR / f'{SYMBOL}_{tf_name}_Dukascopy.parquet'
    if out_path.exists():
        df = pd.read_parquet(out_path)
        min_dt = datetime.fromtimestamp(int(df['time'].min()), tz=timezone.utc)
        max_dt = datetime.fromtimestamp(int(df['time'].max()), tz=timezone.utc)
        log(f'  {tf_name} 完成: {len(df)} 根, {min_dt.strftime("%Y-%m-%d")} → {max_dt.strftime("%Y-%m-%d")}, '
            f'耗时 {elapsed/60:.1f}min, 输出: {out_path.name}')
    else:
        log(f'  {tf_name} 完成但无输出文件')

# ── 用 TV 补齐最新数据 ──
log('\n--- 用 TradingView 数据补齐最新 ---')
for tf_name, network_file in [
    ('H4', f'{SYMBOL}_H4_network.parquet'),
    ('M30', f'{SYMBOL}_M30_network.parquet'),
    ('M15', f'{SYMBOL}_M15_network.parquet'),
    ('M5', None),
]:
    duk_path = CACHE_DIR / f'{SYMBOL}_{tf_name}_Dukascopy.parquet'
    if not duk_path.exists():
        log(f'  {tf_name}: Dukascopy 文件不存在，跳过')
        continue
    if network_file is None:
        log(f'  {tf_name}: 无 TV 文件，跳过')
        continue
    net_path = CACHE_DIR / network_file
    if not net_path.exists():
        log(f'  {tf_name}: TV 文件不存在，跳过')
        continue
    try:
        net = pd.read_parquet(net_path)
        duk = pd.read_parquet(duk_path)
        duk_max = int(duk['time'].max())
        patch = net[net['time'] > duk_max]
        if len(patch) == 0:
            log(f'  {tf_name}: TV 无更新')
            continue
        merged = pd.concat([duk, patch], ignore_index=True)
        merged = merged.sort_values('time').drop_duplicates('time', keep='first').reset_index(drop=True)
        merged.to_parquet(duk_path, index=False)
        log(f'  {tf_name}: 补齐 {len(patch)} 根, 总 {len(merged)} 根')
    except Exception as e:
        log(f'  {tf_name}: TV 补齐失败: {e}')

# ── 最终统计 ──
log('\n=== 最终统计 ===')
for tf_name, _, _ in TIMEFRAMES:
    path = CACHE_DIR / f'{SYMBOL}_{tf_name}_Dukascopy.parquet'
    if path.exists():
        df = pd.read_parquet(path)
        min_dt = datetime.fromtimestamp(int(df['time'].min()), tz=timezone.utc).strftime('%Y-%m-%d')
        max_dt = datetime.fromtimestamp(int(df['time'].max()), tz=timezone.utc).strftime('%Y-%m-%d')
        size = path.stat().st_size / 1e6
        log(f'  {tf_name}: {len(df):>8} 根, {min_dt} → {max_dt}, {size:.2f}MB')
    else:
        log(f'  {tf_name}: 文件不存在')

log(f'\n=== 全部完成 ===')
log(f'结束时间: {datetime.now()}')
