import sys, time, logging, json, urllib.request
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, 'src')

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

CACHE_DIR = Path(r'D:\backup\BaoBao\PythonProgram\miaosuan\data\cache')
CACHE_DIR.mkdir(parents=True, exist_ok=True)
PROXY = 'http://127.0.0.1:10808'
SYMBOL = 'XAUUSD'

# 输出文件
OUT_H4 = CACHE_DIR / f'{SYMBOL}_H4_ultimate.parquet'
OUT_M5 = CACHE_DIR / f'{SYMBOL}_M5_ultimate.parquet'
OUT_M15 = CACHE_DIR / f'{SYMBOL}_M15_ultimate.parquet'
OUT_M30 = CACHE_DIR / f'{SYMBOL}_M30_ultimate.parquet'

# 日志文件
LOG_FILE = CACHE_DIR / 'build_all_timeframes.log'


def log_to_file(msg):
    """同时输出到控制台和日志文件"""
    ts = datetime.now().strftime('%H:%M:%S')
    line = f'{ts} {msg}'
    log.info(msg)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """标准 OHLCV 重采样"""
    df = df.copy()
    df['dt'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df = df.set_index('dt').sort_index()

    agg = df.resample(rule).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'tick_volume': 'sum',
    })
    agg = agg.dropna(subset=['open'])
    agg = agg.reset_index()
    agg['time'] = (agg['dt'].astype('int64') // 1_000_000_000).astype('int64')
    agg = agg.drop(columns=['dt'])

    cols = ['time', 'open', 'high', 'low', 'close', 'volume', 'tick_volume']
    return agg[cols].reset_index(drop=True)


def verify_continuity(df: pd.DataFrame, name: str, expected_secs: int):
    """验证数据连续性"""
    df = df.sort_values('time').reset_index(drop=True)
    diffs = df['time'].diff().dropna().astype(int)

    # H4: 正常 14400s, 周末 3×14400=43200s
    # M30: 正常 1800s, 隔夜 360×1800=648000s(约24h), 周末略多
    # M15: 正常 900s, 隔夜 960×1800=... 更复杂
    # M5: 正常 300s, 隔夜 960×300=288000s
    # 对于非 H4 的周期，只检查极端异常（>2倍正常间隔）
    if name == 'H4':
        expected = {14400, 28800, 43200}  # 1x, 2x, 3x
    elif name == 'M30':
        expected = {1800, 3600, 7200, 14400, 18000, 28800}  # 1x-16x
    elif name == 'M15':
        expected = {900, 1800, 3600, 7200, 14400, 21600}
    elif name == 'M5':
        expected = {300, 600, 1200, 1800, 2400, 3600, 7200, 10800, 14400}
    else:
        expected = {expected_secs}

    unexpected = diffs[~diffs.isin(expected) & (diffs < expected_secs * 5)]
    log_to_file(f'{name}: {len(df)} 根，异常间隙 {len(unexpected)} 个')
    if len(unexpected) > 0:
        log_to_file(f'  异常样例: {unexpected.head(5).tolist()}')
    return len(unexpected)


def step1_generate_h4():
    """步骤1: H1 → H4 重采样（秒级完成）"""
    log_to_file('=' * 60)
    log_to_file('步骤1: H1 → H4 重采样')
    log_to_file('=' * 60)

    h1_path = CACHE_DIR / f'{SYMBOL}_H1_Dukascopy_full.parquet'
    h1 = pd.read_parquet(h1_path)
    log_to_file(f'读取 H1: {len(h1)} 根')

    h4 = resample_ohlcv(h1, '4h')
    tmp = OUT_H4.with_suffix('.tmp')
    h4.to_parquet(tmp, index=False)
    tmp.replace(OUT_H4)

    min_dt = datetime.fromtimestamp(int(h4['time'].min()), tz=timezone.utc).strftime('%Y-%m-%d %H:%M')
    max_dt = datetime.fromtimestamp(int(h4['time'].max()), tz=timezone.utc).strftime('%Y-%m-%d %H:%M')
    log_to_file(f'H4 生成: {len(h4)} 根, {min_dt} → {max_dt}')
    log_to_file(f'  输出: {OUT_H4}')
    verify_continuity(h4, 'H4', 14400)


def step2_fetch_m5():
    """步骤2: 从 Dukascopy 拉取 M5（长时间任务）"""
    log_to_file('=' * 60)
    log_to_file('步骤2: 从 Dukascopy 拉取 M5')
    log_to_file('=' * 60)

    from miaosuan.data.fetchers.dukascopy import DukascopyFetcher

    # 20 年窗口
    window_days = 365 * 20
    fetcher = DukascopyFetcher(proxy=PROXY, timeout=60, window_days=window_days)

    t0 = time.time()
    log_to_file('开始拉取 M5 (20年窗口，约5200个交易日)...')

    # 使用 _fetch_raw 直接获取原始数据
    duk_symbol = fetcher._symbol_to_duk(SYMBOL)
    raw = fetcher._fetch_raw(SYMBOL, 300)  # 300 = M5 秒数

    elapsed = time.time() - t0
    log_to_file(f'M5 拉取完成: {len(raw)} 根, 耗时 {elapsed:.1f}s ({elapsed/60:.1f}分钟)')

    if not raw:
        log_to_file('ERROR: 无数据返回')
        return

    # 构建 DataFrame
    df = pd.DataFrame(raw)
    df = fetcher._finalize(df)
    log_to_file(f'M5 标准化后: {len(df)} 根')

    # 写入缓存
    tmp = OUT_M5.with_suffix('.tmp')
    df.to_parquet(tmp, index=False)
    tmp.replace(OUT_M5)

    min_dt = datetime.fromtimestamp(int(df['time'].min()), tz=timezone.utc).strftime('%Y-%m-%d %H:%M')
    max_dt = datetime.fromtimestamp(int(df['time'].max()), tz=timezone.utc).strftime('%Y-%m-%d %H:%M')
    log_to_file(f'M5 输出: {OUT_M5}')
    log_to_file(f'  范围: {min_dt} → {max_dt}')
    verify_continuity(df, 'M5', 300)


def step3_generate_m15_m30():
    """步骤3: M5 → M15, M30 重采样"""
    log_to_file('=' * 60)
    log_to_file('步骤3: M5 → M15, M30 重采样')
    log_to_file('=' * 60)

    m5 = pd.read_parquet(OUT_M5)
    log_to_file(f'读取 M5: {len(m5)} 根')

    # M15
    m15 = resample_ohlcv(m5, '15min')
    tmp = OUT_M15.with_suffix('.tmp')
    m15.to_parquet(tmp, index=False)
    tmp.replace(OUT_M15)
    min_dt = datetime.fromtimestamp(int(m15['time'].min()), tz=timezone.utc).strftime('%Y-%m-%d %H:%M')
    max_dt = datetime.fromtimestamp(int(m15['time'].max()), tz=timezone.utc).strftime('%Y-%m-%d %H:%M')
    log_to_file(f'M15 生成: {len(m15)} 根, {min_dt} → {max_dt}')
    verify_continuity(m15, 'M15', 900)

    # M30
    m30 = resample_ohlcv(m5, '30min')
    tmp = OUT_M30.with_suffix('.tmp')
    m30.to_parquet(tmp, index=False)
    tmp.replace(OUT_M30)
    min_dt = datetime.fromtimestamp(int(m30['time'].min()), tz=timezone.utc).strftime('%Y-%m-%d %H:%M')
    max_dt = datetime.fromtimestamp(int(m30['time'].max()), tz=timezone.utc).strftime('%Y-%m-%d %H:%M')
    log_to_file(f'M30 生成: {len(m30)} 根, {min_dt} → {max_dt}')
    verify_continuity(m30, 'M30', 1800)


def step4_append_latest_tv():
    """步骤4: 用 TV 数据补齐 2026-08-31 之后的最新数据"""
    log_to_file('=' * 60)
    log_to_file('步骤4: 用 TV 数据补齐最新')
    log_to_file('=' * 60)

    # H4: 用 H4_network 补齐
    for tf, network_file, ultimate_file in [
        ('H4', f'{SYMBOL}_H4_network.parquet', f'{SYMBOL}_H4_ultimate.parquet'),
        ('M30', f'{SYMBOL}_M30_network.parquet', f'{SYMBOL}_M30_ultimate.parquet'),
        ('M15', f'{SYMBOL}_M15_network.parquet', f'{SYMBOL}_M15_ultimate.parquet'),
    ]:
        net_path = CACHE_DIR / network_file
        ult_path = CACHE_DIR / ultimate_file

        if not net_path.exists() or not ult_path.exists():
            log_to_file(f'{tf}: 跳过 (文件不存在)')
            continue

        net = pd.read_parquet(net_path)
        ult = pd.read_parquet(ult_path)

        ult_max_ts = int(ult['time'].max())
        patch = net[net['time'] > ult_max_ts]

        if len(patch) == 0:
            log_to_file(f'{tf}: 无需补齐 (TV 无更新数据)')
            continue

        merged = pd.concat([ult, patch], ignore_index=True)
        merged = merged.sort_values('time').reset_index(drop=True)
        merged = merged[~merged['time'].duplicated(keep='first')].reset_index(drop=True)

        cols = ['time', 'open', 'high', 'low', 'close', 'volume', 'tick_volume']
        merged = merged[cols]

        tmp = ult_path.with_suffix('.tmp')
        merged.to_parquet(tmp, index=False)
        tmp.replace(ult_path)

        log_to_file(f'{tf}: 补齐 {len(patch)} 根, 总 {len(merged)} 根')
        log_to_file(f'  最新: {datetime.fromtimestamp(int(merged["time"].max()), tz=timezone.utc).strftime("%Y-%m-%d %H:%M")}')


def main():
    # 清空日志
    open(LOG_FILE, 'w').close()

    log_to_file('XAUUSD 全周期数据集构建开始')
    log_to_file(f'开始时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

    # 步骤1: H4 (秒级)
    step1_generate_h4()

    # 步骤2: M5 (长时间)
    step2_fetch_m5()

    # 步骤3: M15, M30 (秒级)
    step3_generate_m15_m30()

    # 步骤4: TV 补齐
    step4_append_latest_tv()

    log_to_file('=' * 60)
    log_to_file('全部完成!')
    log_to_file(f'结束时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    log_to_file('=' * 60)

    # 最终统计
    log_to_file('\n=== 最终数据集统计 ===')
    for name, path in [
        ('H4', OUT_H4), ('M5', OUT_M5), ('M15', OUT_M15), ('M30', OUT_M30)
    ]:
        if path.exists():
            df = pd.read_parquet(path)
            min_dt = datetime.fromtimestamp(int(df['time'].min()), tz=timezone.utc).strftime('%Y-%m-%d')
            max_dt = datetime.fromtimestamp(int(df['time'].max()), tz=timezone.utc).strftime('%Y-%m-%d')
            log_to_file(f'{name}: {len(df):>8} 根, {min_dt} → {max_dt}')


if __name__ == '__main__':
    main()