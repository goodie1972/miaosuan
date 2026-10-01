# -*- coding: utf-8 -*-
"""
XAUUSD 完整数据合成：Dukascopy H1 全量拉取 → 本地重采样 D1/W1/MN1
解决「最长历史」与「最新数据」无法兼得的矛盾。

使用第一性原理：
- Dukascopy D1 按年请求 → 当年 400（未完年）
- Dukascopy H1 按月请求 → 已完月份全有
- 本地重采样：H1 → D1/W1/MN1（pandas resample，OHLCV 严格正确）
"""

import sys
import time
import logging
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, 'src')
import pandas as pd
import numpy as np

from miaosuan.data.fetchers.dukascopy import DukascopyFetcher
from miaosuan.data.acquisition import DataAcquisition, _slugify

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

CACHE_DIR = Path(r'D:\backup\BaoBao\PythonProgram\miaosuan\data\cache')
CACHE_DIR.mkdir(parents=True, exist_ok=True)

PROXY = 'http://127.0.0.1:10808'
SYMBOL = 'XAUUSD'

# 目标输出文件
OUT_H1 = CACHE_DIR / f'{SYMBOL}_H1_Dukascopy_full.parquet'
OUT_D1 = CACHE_DIR / f'{SYMBOL}_D1_Dukascopy_full.parquet'
OUT_W1 = CACHE_DIR / f'{SYMBOL}_W1_Dukascopy_full.parquet'
OUT_MN1 = CACHE_DIR / f'{SYMBOL}_MN1_Dukascopy_full.parquet'


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    标准 OHLCV 重采样（pandas resample）。
    rule: 'D'=日, 'W-MON'=周一开盘, 'MS'=月首
    """
    # 确保 time 是 datetime index（UTC）
    df = df.copy()
    df['dt'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df = df.set_index('dt').sort_index()
    
    # 重采样聚合
    agg = df.resample(rule).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'tick_volume': 'sum',
    })
    agg = agg.dropna(subset=['open'])  # 删除无数据的周期
    
    # 恢复 time 列（周期开始时间的 Unix 秒）
    agg = agg.reset_index()
    agg['time'] = (agg['dt'].astype('int64') // 1_000_000_000).astype('int64')
    agg = agg.drop(columns=['dt'])
    
    # 标准列顺序
    cols = ['time', 'open', 'high', 'low', 'close', 'volume', 'tick_volume']
    return agg[cols].reset_index(drop=True)


def fetch_h1_full() -> pd.DataFrame:
    """拉取 Dukascopy H1 全量（20 年窗口）"""
    log.info(f'开始拉取 {SYMBOL} H1 全量数据...')
    fetcher = DukascopyFetcher(proxy=PROXY, timeout=60, window_days=365*20)
    
    t0 = time.time()
    df = fetcher.fetch_full(SYMBOL, 'H1')
    elapsed = time.time() - t0
    
    log.info(f'H1 全量拉取完成：{len(df)} 根，耗时 {elapsed:.1f}s')
    log.info(f'时间范围：{datetime.fromtimestamp(int(df["time"].min()), tz=timezone.utc)} → '
             f'{datetime.fromtimestamp(int(df["time"].max()), tz=timezone.utc)}')
    
    # 原子写入缓存
    OUT_H1.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT_H1.with_suffix('.tmp')
    df.to_parquet(tmp, index=False)
    tmp.replace(OUT_H1)
    log.info(f'H1 缓存已写入：{OUT_H1}')
    
    return df


def build_all_timeframes(h1_df: pd.DataFrame):
    """从 H1 重采样生成 D1/W1/MN1"""
    log.info('开始重采样生成 D1/W1/MN1...')
    
    # D1
    d1 = resample_ohlcv(h1_df, 'D')
    tmp = OUT_D1.with_suffix('.tmp')
    d1.to_parquet(tmp, index=False)
    tmp.replace(OUT_D1)
    log.info(f'D1 生成：{len(d1)} 根 → {OUT_D1}')
    log.info(f'  范围：{datetime.fromtimestamp(int(d1["time"].min()), tz=timezone.utc)} → '
             f'{datetime.fromtimestamp(int(d1["time"].max()), tz=timezone.utc)}')
    
    # W1（周一开盘）
    w1 = resample_ohlcv(h1_df, 'W-MON')
    tmp = OUT_W1.with_suffix('.tmp')
    w1.to_parquet(tmp, index=False)
    tmp.replace(OUT_W1)
    log.info(f'W1 生成：{len(w1)} 根 → {OUT_W1}')
    
    # MN1（月首）
    mn1 = resample_ohlcv(h1_df, 'MS')
    tmp = OUT_MN1.with_suffix('.tmp')
    mn1.to_parquet(tmp, index=False)
    tmp.replace(OUT_MN1)
    log.info(f'MN1 生成：{len(mn1)} 根 → {OUT_MN1}')
    
    return d1, w1, mn1


def verify_continuity():
    """验证生成数据的连续性"""
    log.info('=== 连续性验证 ===')
    for name, path in [('H1', OUT_H1), ('D1', OUT_D1), ('W1', OUT_W1), ('MN1', OUT_MN1)]:
        df = pd.read_parquet(path)
        df = df.sort_values('time').reset_index(drop=True)
        
        # 检查时间间隔
        expected_sec = {'H1': 3600, 'D1': 86400, 'W1': 604800, 'MN1': 2592000}[name]
        diffs = df['time'].diff().dropna().astype(int)
        
        # 允许周末/节假日间隙，只检查非预期间隙
        if name == 'H1':
            # H1 工作日应连续，周末允许 48h/72h 间隙
            unexpected = diffs[(diffs != 3600) & (diffs != 7200) & (diffs != 10800) & (diffs < 3600*24*5)]
        elif name == 'D1':
            # D1 交易日应连续（跳过周末）
            unexpected = diffs[(diffs != 86400) & (diffs != 172800) & (diffs != 259200)]
        else:
            unexpected = pd.Series([], dtype=int)
        
        log.info(f'{name}: {len(df)} 根，异常间隙 {len(unexpected)} 个，'
                 f'最早 {datetime.fromtimestamp(int(df["time"].min()), tz=timezone.utc)}, '
                 f'最新 {datetime.fromtimestamp(int(df["time"].max()), tz=timezone.utc)}')
        
        if len(unexpected) > 0:
            log.warning(f'  异常间隙样例: {unexpected.head(5).tolist()}')


def main():
    log.info('=' * 60)
    log.info('XAUUSD 完整数据合成开始')
    log.info('=' * 60)
    
    # 1. 拉取 H1 全量（如已有缓存可跳过）
    if OUT_H1.exists():
        log.info(f'发现已有 H1 缓存：{OUT_H1}，读取...')
        h1_df = pd.read_parquet(OUT_H1)
        log.info(f'读取 H1：{len(h1_df)} 根')
    else:
        h1_df = fetch_h1_full()
    
    # 2. 重采样生成所有周期
    d1, w1, mn1 = build_all_timeframes(h1_df)
    
    # 3. 验证
    verify_continuity()
    
    log.info('=' * 60)
    log.info('完成！输出文件：')
    log.info(f'  H1:  {OUT_H1}')
    log.info(f'  D1:  {OUT_D1}')
    log.info(f'  W1:  {OUT_W1}')
    log.info(f'  MN1: {OUT_MN1}')
    log.info('=' * 60)


if __name__ == '__main__':
    main()