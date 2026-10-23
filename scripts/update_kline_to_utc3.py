#!/usr/bin/env python3
"""
更新 XAUUSD K线数据到最新，并转换为 UTC+3 时间戳
"""

import sys
sys.path.insert(0, 'src')

import pandas as pd
import os
from datetime import datetime, timezone
from miaosuan.data.fetchers.dukascopy import DukascopyFetcher

CACHE_DIR = 'data/cache'
BACKUP_DIR = 'tmp/data_cache_backup'

# 目标截止日期
TARGET_DATE = '2026-10-23'
TARGET_TS = int(datetime(2026, 10, 23, 23, 59, 59, tzinfo=timezone.utc).timestamp())

# UTC+3 偏移量（秒）
UTC3_OFFSET = 10800

TIMEFRAMES = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4']

def update_timeframe(fetcher, tf, since_ts):
    """获取指定周期的增量数据"""
    print(f"\n{'='*60}")
    print(f"更新 {tf} 从 {datetime.fromtimestamp(since_ts, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"{'='*60}")
    
    try:
        df_new = fetcher.fetch_incremental('XAUUSD', tf, since_ts)
        
        if df_new is None or len(df_new) == 0:
            print(f"  无新数据")
            return None
        
        print(f"  获取到 {len(df_new):,} 根新 K线")
        print(f"  时间范围: {pd.to_datetime(df_new['time'].min(), unit='s', utc=True)} ~ "
              f"{pd.to_datetime(df_new['time'].max(), unit='s', utc=True)}")
        
        return df_new
    except Exception as e:
        print(f"  X 获取失败: {e}")
        return None

def convert_to_utc3(df):
    """将 UTC 时间戳转换为 UTC+3"""
    df = df.copy()
    df['time'] = df['time'] + UTC3_OFFSET
    return df

def main():
    print("=" * 80)
    print("XAUUSD K线数据更新脚本")
    print(f"目标日期: {TARGET_DATE}")
    print(f"时间戳转换: UTC → UTC+3 (+{UTC3_OFFSET}s)")
    print("=" * 80)
    
    fetcher = DukascopyFetcher()
    
    # 检查 fetcher 是否可用
    if not fetcher.is_available():
        print("X DukascopyFetcher 不可用")
        return
    
    print(f"\nDukascopy 支持的周期: {fetcher.supported_timeframes()}")
    
    results = {}
    
    for tf in TIMEFRAMES:
        parquet_path = os.path.join(CACHE_DIR, f'XAUUSD_{tf}.parquet')
        
        if not os.path.exists(parquet_path):
            print(f"\nX {tf}: 文件不存在，跳过")
            continue
        
        # 读取现有数据
        df_existing = pd.read_parquet(parquet_path)
        existing_max_ts = df_existing['time'].max()
        
        print(f"\n{tf}: 现有数据截止 {pd.to_datetime(existing_max_ts, unit='s', utc=True).strftime('%Y-%m-%d %H:%M')} UTC")
        
        # 获取增量数据
        df_new = update_timeframe(fetcher, tf, existing_max_ts)
        
        if df_new is not None and len(df_new) > 0:
            # 合并数据
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            
            # 去重
            df_combined = df_combined.drop_duplicates(subset=['time'], keep='last')
            
            # 按时间排序
            df_combined = df_combined.sort_values('time').reset_index(drop=True)
            
            # 转换为 UTC+3
            df_utc3 = convert_to_utc3(df_combined)
            
            # 保存
            backup_path = os.path.join(BACKUP_DIR, f'XAUUSD_{tf}_before_update.parquet')
            os.makedirs(BACKUP_DIR, exist_ok=True)
            df_existing.to_parquet(backup_path, index=False)
            print(f"  备份到: {backup_path}")
            
            df_utc3.to_parquet(parquet_path, index=False)
            print(f"  [OK] 已更新并转换为 UTC+3: {parquet_path}")
            print(f"  总行数: {len(df_utc3):,}")
            print(f"  新截止时间: {pd.to_datetime(df_utc3['time'].max() - UTC3_OFFSET, unit='s', utc=True).strftime('%Y-%m-%d %H:%M')} UTC")
            
            results[tf] = {
                'old_count': len(df_existing),
                'new_count': len(df_new),
                'total_count': len(df_utc3),
                'new_end_utc': pd.to_datetime(df_utc3['time'].max() - UTC3_OFFSET, unit='s', utc=True)
            }
        else:
            print(f"  [!] 无新数据，保持原样")
            results[tf] = {
                'old_count': len(df_existing),
                'new_count': 0,
                'total_count': len(df_existing),
                'new_end_utc': pd.to_datetime(existing_max_ts, unit='s', utc=True)
            }
    
    # 汇总报告
    print("\n" + "=" * 80)
    print("更新完成汇总")
    print("=" * 80)
    print(f"\n{'周期':<8s} {'原有':>10s} {'新增':>10s} {'总计':>10s} {'截止(UTC)':>20s}")
    print("-" * 70)
    
    for tf, info in results.items():
        print(f"{tf:<8s} {info['old_count']:>10,} {info['new_count']:>10,} {info['total_count']:>10,} "
              f"{info['new_end_utc'].strftime('%Y-%m-%d %H:%M'):>20s}")
    
    print("\n[OK] 所有数据已更新并转换为 UTC+3 时间戳")

if __name__ == '__main__':
    main()
