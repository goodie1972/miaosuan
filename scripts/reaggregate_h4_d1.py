#!/usr/bin/env python3
"""
从 M1 数据重新聚合 H4/D1，使用实盘经纪商的桶边界

桶原点:
  H4: 01:00/05:00/09:00/13:00/17:00/21:00 UTC+3 (time % 14400 == 3600)
  D1: 00:00 UTC+3 (time % 86400 == 0)
"""

import sys
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
import os

CACHE_DIR = 'data/cache'

# UTC+3 偏移量（已在数据中）
UTC3_OFFSET = 10800

# 桶边界偏移
H4_OFFSET = 3600   # 01:00 UTC+3
D1_OFFSET = 0      # 00:00 UTC+3

def aggregate_from_m1(tf, offset):
    """从 M1 数据聚合指定周期"""
    print(f"\n{'='*60}")
    print(f"从 M1 聚合 {tf}")
    print(f"桶边界偏移: {offset}s ({offset/3600:.1f}h)")
    print(f"{'='*60}")
    
    # 读取 M1 数据
    m1_path = os.path.join(CACHE_DIR, 'XAUUSD_M1.parquet')
    df_m1 = pd.read_parquet(m1_path)
    print(f"M1 数据: {len(df_m1):,} 根")
    
    # 计算桶边界
    tf_seconds = {'H4': 14400, 'D1': 86400}[tf]
    
    # bucket = time - ((time - offset) % tf_seconds)
    df_m1['bucket'] = df_m1['time'] - ((df_m1['time'] - offset) % tf_seconds)
    
    # 聚合
    agg = df_m1.groupby('bucket').agg(
        time=('bucket', 'first'),
        open=('open', 'first'),
        high=('high', 'max'),
        low=('low', 'min'),
        close=('close', 'last'),
        volume=('volume', 'sum'),
        count=('close', 'count')
    ).reset_index(drop=True)
    
    print(f"聚合后: {len(agg):,} 根")
    
    # 验证桶边界
    mod = agg['time'] % tf_seconds
    print(f"桶边界验证 (time % {tf_seconds}):")
    print(mod.value_counts().sort_index())
    
    # 过滤不完整的 bar（count 不足）
    expected_count = {'H4': 240, 'D1': 1440}[tf]  # M1 bars per H4/D1
    min_count = int(expected_count * 0.8)  # 至少 80% 的 bar
    
    incomplete = (agg['count'] < min_count).sum()
    print(f"不完整 bar (< {min_count} M1 bars): {incomplete:,}")
    
    # 保留不完整的 bar（可能包含部分交易时段）
    # 但标记出来
    agg['complete'] = agg['count'] >= min_count
    
    print(f"完整 bar: {agg['complete'].sum():,}")
    print(f"不完整 bar: {(~agg['complete']).sum():,}")
    
    # 删除 count 和 complete 列
    agg = agg.drop(columns=['count', 'complete'])
    
    # 时间范围
    agg['dt'] = pd.to_datetime(agg['time'], unit='s', utc=True)
    print(f"时间范围: {agg['dt'].min()} ~ {agg['dt'].max()}")
    
    return agg

def main():
    print("=" * 80)
    print("从 M1 重新聚合 H4/D1")
    print("使用实盘经纪商桶边界")
    print("=" * 80)
    
    # 聚合 H4
    df_h4 = aggregate_from_m1('H4', H4_OFFSET)
    h4_path = os.path.join(CACHE_DIR, 'XAUUSD_H4.parquet')
    df_h4.to_parquet(h4_path, index=False)
    print(f"\n[OK] 已保存: {h4_path}")
    
    # 聚合 D1
    df_d1 = aggregate_from_m1('D1', D1_OFFSET)
    d1_path = os.path.join(CACHE_DIR, 'XAUUSD_D1.parquet')
    df_d1.to_parquet(d1_path, index=False)
    print(f"\n[OK] 已保存: {d1_path}")
    
    # 验证
    print("\n" + "=" * 80)
    print("验证")
    print("=" * 80)
    
    for tf, path in [('H4', h4_path), ('D1', d1_path)]:
        df = pd.read_parquet(path)
        tf_seconds = {'H4': 14400, 'D1': 86400}[tf]
        offset = {'H4': H4_OFFSET, 'D1': D1_OFFSET}[tf]
        
        mod = df['time'] % tf_seconds
        print(f"\n{tf}:")
        print(f"  行数: {len(df):,}")
        print(f"  桶边界 (time % {tf_seconds}): {mod.unique()}")
        print(f"  期望偏移: {offset}")
        print(f"  匹配: {len(mod.unique()) == 1 and mod.iloc[0] == offset}")
    
    print("\n[OK] H4/D1 已重新聚合，桶边界与实盘一致")

if __name__ == '__main__':
    main()
