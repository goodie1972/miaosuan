#!/usr/bin/env python3
"""
从 ShenjiDB 导出所有数据并转换为 UTC+3 时间戳
"""

import sys
sys.path.insert(0, 'src')

import pandas as pd
import os
from datetime import datetime, timezone

CACHE_DIR = 'data/cache'
DB_PATH = 'D:/backup/BaoBao/PythonProgram/xauusd/data/market_data.db'

# UTC+3 偏移量（秒）
UTC3_OFFSET = 10800

TIMEFRAMES = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1']

def main():
    print("=" * 80)
    print("从 ShenjiDB 导出数据并转换为 UTC+3")
    print(f"数据库: {DB_PATH}")
    print("=" * 80)
    
    from miaosuan.data.fetchers.shenji_db import ShenjiDBFetcher
    
    if not os.path.exists(DB_PATH):
        print(f"X 数据库不存在: {DB_PATH}")
        return
    
    fetcher = ShenjiDBFetcher(DB_PATH)
    
    if not fetcher.is_available():
        print("X ShenjiDBFetcher 不可用")
        return
    
    print(f"\n支持的周期: {fetcher.supported_timeframes()}")
    
    results = {}
    
    for tf in TIMEFRAMES:
        print(f"\n{'='*60}")
        print(f"导出 {tf}")
        print(f"{'='*60}")
        
        try:
            # 获取完整数据
            df = fetcher.fetch_full('XAUUSD', tf)
            
            if df is None or len(df) == 0:
                print(f"  X 无数据")
                continue
            
            print(f"  获取到 {len(df):,} 根 K线")
            
            # 检查时间戳
            df['dt_utc'] = pd.to_datetime(df['time'], unit='s', utc=True)
            print(f"  原始时间范围 (UTC): {df['dt_utc'].min()} ~ {df['dt_utc'].max()}")
            
            # 转换为 UTC+3
            df['time_utc3'] = df['time'] + UTC3_OFFSET
            df['dt_utc3'] = pd.to_datetime(df['time_utc3'], unit='s', utc=True)
            print(f"  UTC+3 时间范围: {df['dt_utc3'].min()} ~ {df['dt_utc3'].max()}")
            
            # 准备输出
            df_out = df[['time_utc3', 'open', 'high', 'low', 'close', 'volume']].copy()
            df_out.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
            df_out = df_out.sort_values('time').reset_index(drop=True)
            
            # 保存
            output_path = os.path.join(CACHE_DIR, f'XAUUSD_{tf}_shenji_utc3.parquet')
            df_out.to_parquet(output_path, index=False)
            print(f"  [OK] 已保存: {output_path}")
            print(f"  行数: {len(df_out):,}")
            
            results[tf] = {
                'count': len(df_out),
                'start_utc3': df_out['time'].min(),
                'end_utc3': df_out['time'].max(),
                'start_dt': pd.to_datetime(df_out['time'].min() - UTC3_OFFSET, unit='s', utc=True),
                'end_dt': pd.to_datetime(df_out['time'].max() - UTC3_OFFSET, unit='s', utc=True),
            }
            
        except Exception as e:
            print(f"  X 错误: {e}")
            import traceback
            traceback.print_exc()
    
    # 汇总报告
    print("\n" + "=" * 80)
    print("导出完成汇总")
    print("=" * 80)
    print(f"\n{'周期':<8s} {'行数':>10s} {'起始(UTC)':>20s} {'截止(UTC)':>20s}")
    print("-" * 70)
    
    for tf, info in results.items():
        print(f"{tf:<8s} {info['count']:>10,} "
              f"{info['start_dt'].strftime('%Y-%m-%d %H:%M'):>20s} "
              f"{info['end_dt'].strftime('%Y-%m-%d %H:%M'):>20s}")
    
    print("\n[OK] 所有数据已导出并转换为 UTC+3 时间戳")
    print(f"\n输出文件: {CACHE_DIR}/XAUUSD_*_shenji_utc3.parquet")

if __name__ == '__main__':
    main()
