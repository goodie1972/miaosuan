#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MT4 导出的 CSV → Parquet（符合 MiaoSuan 期望的 Schema）"""

import pandas as pd
import sys
from pathlib import Path

def csv_to_parquet(csv_path: str, out_path: str = None):
    """转换 MT4 CSV 为 MiaoSuan 兼容的 Parquet
    
    MT4 CSV 列：time(Unix秒),open,high,low,close,volume,tick_volume
    目标 Parquet 列：time(int64秒),open,high,low,close,volume,tick_volume (float64)
    """
    df = pd.read_csv(csv_path)
    
    # 校验列名
    expected = ['time','open','high','low','close','volume','tick_volume']
    if list(df.columns) != expected:
        raise ValueError(f"列名不匹配: 期望 {expected}, 实际 {list(df.columns)}")
    
    # 类型转换
    df['time'] = df['time'].astype('int64')  # Unix 秒
    for col in ['open','high','low','close']:
        df[col] = df[col].astype('float64')
    df['volume'] = df['volume'].astype('float64')
    df['tick_volume'] = df['tick_volume'].astype('float64')
    
    # 排序 & 去重
    df = df.sort_values('time').drop_duplicates(subset='time', keep='last')
    
    # 输出
    out = Path(out_path) if out_path else Path(csv_path).with_suffix('.parquet')
    df.to_parquet(out, index=False)
    print(f"OK: {len(df)} bars -> {out}")
    print(f"Range: {pd.Timestamp(df['time'].iloc[0], unit='s')} -> {pd.Timestamp(df['time'].iloc[-1], unit='s')}")
    return out

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python csv_to_parquet.py <input.csv> [output.parquet]")
        sys.exit(1)
    csv_to_parquet(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)