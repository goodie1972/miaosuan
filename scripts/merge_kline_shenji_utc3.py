#!/usr/bin/env python3
"""
合并 kline (2006-2026-10-01) 和 ShenjiDB (近期) 数据
统一转换为 UTC+3 时间戳
"""

import sys
sys.path.insert(0, 'src')

import pandas as pd
import os
from datetime import datetime, timezone

CACHE_DIR = 'data/cache'
BACKUP_DIR = 'tmp/data_cache_backup'

# UTC+3 偏移量（秒）
UTC3_OFFSET = 10800

TIMEFRAMES = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1']

def merge_timeframes(tf):
    """合并指定周期的 kline 和 shenji 数据"""
    print(f"\n{'='*60}")
    print(f"合并 {tf}")
    print(f"{'='*60}")
    
    kline_path = os.path.join(CACHE_DIR, f'XAUUSD_{tf}.parquet')
    shenji_path = os.path.join(CACHE_DIR, f'XAUUSD_{tf}_shenji_utc3.parquet')
    output_path = os.path.join(CACHE_DIR, f'XAUUSD_{tf}_merged_utc3.parquet')
    
    # 读取 kline 数据（UTC）
    if os.path.exists(kline_path):
        df_kline = pd.read_parquet(kline_path)
        df_kline['source'] = 'kline'
        # 转换为 UTC+3
        df_kline['time'] = df_kline['time'] + UTC3_OFFSET
        print(f"  kline: {len(df_kline):,} 根")
        print(f"    时间范围 (UTC+3): {pd.to_datetime(df_kline['time'].min(), unit='s', utc=True)} ~ "
              f"{pd.to_datetime(df_kline['time'].max(), unit='s', utc=True)}")
    else:
        df_kline = pd.DataFrame()
        print(f"  kline: 不存在")
    
    # 读取 shenji 数据（已是 UTC+3）
    if os.path.exists(shenji_path):
        df_shenji = pd.read_parquet(shenji_path)
        df_shenji['source'] = 'shenji'
        print(f"  shenji: {len(df_shenji):,} 根")
        print(f"    时间范围 (UTC+3): {pd.to_datetime(df_shenji['time'].min(), unit='s', utc=True)} ~ "
              f"{pd.to_datetime(df_shenji['time'].max(), unit='s', utc=True)}")
    else:
        df_shenji = pd.DataFrame()
        print(f"  shenji: 不存在")
    
    if df_kline.empty and df_shenji.empty:
        print(f"  X 无数据可合并")
        return None
    
    if df_kline.empty:
        df_merged = df_shenji
        print(f"  [!] 仅使用 shenji 数据")
    elif df_shenji.empty:
        df_merged = df_kline
        print(f"  [!] 仅使用 kline 数据")
    else:
        # 合并
        df_merged = pd.concat([df_kline, df_shenji], ignore_index=True)
        
        # 去重（保留 shenji 数据，因为更新）
        df_merged = df_merged.drop_duplicates(subset=['time'], keep='last')
        
        # 按时间排序
        df_merged = df_merged.sort_values('time').reset_index(drop=True)
        
        print(f"  合并后: {len(df_merged):,} 根")
        print(f"    时间范围 (UTC+3): {pd.to_datetime(df_merged['time'].min(), unit='s', utc=True)} ~ "
              f"{pd.to_datetime(df_merged['time'].max(), unit='s', utc=True)}")
    
    # 删除 source 列
    if 'source' in df_merged.columns:
        df_merged = df_merged.drop(columns=['source'])
    
    # 保存
    df_merged.to_parquet(output_path, index=False)
    print(f"  [OK] 已保存: {output_path}")
    
    return {
        'count': len(df_merged),
        'start': pd.to_datetime(df_merged['time'].min(), unit='s', utc=True),
        'end': pd.to_datetime(df_merged['time'].max(), unit='s', utc=True),
    }

def main():
    print("=" * 80)
    print("合并 kline + ShenjiDB 数据")
    print("统一转换为 UTC+3 时间戳")
    print("=" * 80)
    
    results = {}
    
    for tf in TIMEFRAMES:
        result = merge_timeframes(tf)
        if result:
            results[tf] = result
    
    # 汇总报告
    print("\n" + "=" * 80)
    print("合并完成汇总")
    print("=" * 80)
    print(f"\n{'周期':<8s} {'行数':>10s} {'起始(UTC+3)':>25s} {'截止(UTC+3)':>25s}")
    print("-" * 80)
    
    for tf, info in results.items():
        print(f"{tf:<8s} {info['count']:>10,} "
              f"{info['start'].strftime('%Y-%m-%d %H:%M'):>25s} "
              f"{info['end'].strftime('%Y-%m-%d %H:%M'):>25s}")
    
    print("\n[OK] 所有数据已合并并转换为 UTC+3 时间戳")
    print(f"\n输出文件: {CACHE_DIR}/XAUUSD_*_merged_utc3.parquet")

if __name__ == '__main__':
    main()
