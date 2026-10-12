#!/usr/bin/env python3
"""
XAUUSD K线数据清理脚本
过滤陈旧 bar 和周末填充，调整时间戳对齐到 UTC+3
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timezone, timedelta
import argparse


def is_stale_bar(df):
    """检测陈旧 bar：o==h==l==c 且 vol==0"""
    return (df['open'] == df['high']) & \
           (df['high'] == df['low']) & \
           (df['low'] == df['close']) & \
           (df['volume'] == 0)


def is_sunday(df_dt):
    """检测周日 bar（周日=6，周六=5 在黄金市场有交易）"""
    return df_dt.dt.dayofweek == 6


def clean_xauusd_data(input_path, output_path=None, filter_stale=True, filter_weekend=True,
                      align_utc3=False):
    """
    清理 XAUUSD K线数据

    Args:
        input_path: 输入 parquet 文件路径
        output_path: 输出路径（默认覆盖原文件）
        filter_stale: 是否过滤陈旧 bar（o==h==l==c, vol==0）
        filter_weekend: 是否过滤周末 bar
        align_utc3: 是否将时间戳调整为 UTC+3

    Returns:
        清理前后的统计数据
    """
    print(f"读取: {input_path}")
    df = pd.read_parquet(input_path)
    original_len = len(df)

    # 添加时间列用于分析
    df['dt'] = pd.to_datetime(df['time'], unit='s', utc=True)

    # 统计过滤前的数据
    stale_count = is_stale_bar(df).sum() if filter_stale else 0
    weekend_count = is_sunday(df['dt']).sum() if filter_weekend else 0

    # 应用过滤
    mask = pd.Series([True] * len(df), index=df.index)

    if filter_stale:
        stale_mask = is_stale_bar(df)
        mask &= ~stale_mask
        print(f"  陈旧 bar: {stale_count:,} 根")

    if filter_weekend:
        weekend_mask = is_sunday(df['dt'])
        mask &= ~weekend_mask
        print(f"  周末 bar: {weekend_count:,} 根")

    df_clean = df[mask].copy()
    df_clean = df_clean.drop(columns=['dt'])

    # 时间戳调整
    if align_utc3:
        # UTC+3 = 期货交易日标准（纽约夏令时）
        df_clean['time'] = df_clean['time'] + 10800  # +3小时

    # 清理后的统计
    cleaned_len = len(df_clean)
    removed_count = original_len - cleaned_len

    print(f"\n清理结果:")
    print(f"  原始行数: {original_len:,}")
    print(f"  清理后:   {cleaned_len:,}")
    print(f"  移除:     {removed_count:,} ({removed_count/original_len*100:.2f}%)")

    # 时间范围
    if 'time' in df_clean.columns:
        start_dt = pd.to_datetime(df_clean['time'].iloc[0], unit='s', utc=True)
        end_dt = pd.to_datetime(df_clean['time'].iloc[-1], unit='s', utc=True)
        if align_utc3:
            start_dt = start_dt + timedelta(hours=3)
            end_dt = end_dt + timedelta(hours=3)
        print(f"  时间范围: {start_dt.strftime('%Y-%m-%d')} ~ {end_dt.strftime('%Y-%m-%d')}")

    # 保存
    if output_path is None:
        output_path = input_path

    print(f"\n保存: {output_path}")
    df_clean.to_parquet(output_path, index=False)

    return {
        'original': original_len,
        'cleaned': cleaned_len,
        'removed': removed_count,
        'stale': stale_count,
        'weekend': weekend_count,
    }


def main():
    parser = argparse.ArgumentParser(description='清理 XAUUSD K线数据')
    parser.add_argument('--input', '-i', required=True, help='输入 parquet 文件')
    parser.add_argument('--output', '-o', help='输出文件（默认覆盖原文件）')
    parser.add_argument('--no-stale-filter', action='store_true', help='不过滤陈旧 bar')
    parser.add_argument('--no-weekend-filter', action='store_true', help='不过滤周末 bar')
    parser.add_argument('--align-utc3', action='store_true', help='时间戳对齐到 UTC+3')
    args = parser.parse_args()

    result = clean_xauusd_data(
        input_path=args.input,
        output_path=args.output,
        filter_stale=not args.no_stale_filter,
        filter_weekend=not args.no_weekend_filter,
        align_utc3=args.align_utc3,
    )

    print("\n" + "=" * 60)
    print("完成")


if __name__ == '__main__':
    main()
