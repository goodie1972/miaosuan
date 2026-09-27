#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""批量转换 MT4 导出的 CSV → Parquet（MiaoSuan 兼容格式）

用法:
    python scripts/convert_all.py                    # 转换 D:\K线数据 下所有 *_*Y.csv
    python scripts/convert_all.py --dir D:\K线数据    # 指定目录
    python scripts/convert_all.py --years 5           -- 只转换 *5Y.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd


def csv_to_parquet(csv_path: Path) -> Path:
    """MT4 CSV → MiaoSuan Parquet

    CSV 列: time(Unix秒),open,high,low,close,volume,tick_volume
    Parquet 列: time(int64),open,high,low,close,volume,tick_volume (float64)
    """
    df = pd.read_csv(csv_path)

    expected = ["time", "open", "high", "low", "close", "volume", "tick_volume"]
    if list(df.columns) != expected:
        raise ValueError(f"{csv_path.name}: 列名不匹配，期望 {expected}，实际 {list(df.columns)}")

    df["time"] = df["time"].astype("int64")
    for col in ["open", "high", "low", "close", "volume", "tick_volume"]:
        df[col] = df[col].astype("float64")

    df = df.sort_values("time").drop_duplicates(subset="time", keep="last").reset_index(drop=True)

    out = csv_path.with_suffix(".parquet")
    df.to_parquet(out, index=False)

    t0 = pd.Timestamp(df["time"].iloc[0], unit="s")
    t1 = pd.Timestamp(df["time"].iloc[-1], unit="s")
    span = (t1 - t0).days
    print(f"  {csv_path.name:40s} -> {out.name:40s}  {len(df):6d} bars  {t0.date()} -> {t1.date()}  ({span/365.25:.1f}y)")
    return out


def main():
    parser = argparse.ArgumentParser(description="批量转换 MT4 CSV → Parquet")
    parser.add_argument("--dir", default=r"D:\K线数据", help="CSV 所在目录")
    parser.add_argument("--years", type=int, default=None, help="只转换指定年数的文件（如 5）")
    args = parser.parse_args()

    data_dir = Path(args.dir)
    if not data_dir.exists():
        print(f"错误：目录不存在 {data_dir}")
        sys.exit(1)

    # 找所有 CSV
    pattern = "*.csv"
    csvs = sorted(data_dir.glob(pattern))

    # 过滤年数
    if args.years:
        csvs = [c for c in csvs if f"{args.years}Y" in c.name]

    # 排除非 MT4 导出的 CSV（没有标准列头的）
    valid = []
    for c in csvs:
        try:
            head = pd.read_csv(c, nrows=1)
            if "time" in head.columns and "open" in head.columns:
                valid.append(c)
        except Exception:
            pass

    if not valid:
        print(f"未找到有效的 MT4 CSV 文件（{data_dir}）")
        print("请先在 MT4 中运行 scripts/export_all_tf.mq4 导出数据")
        sys.exit(1)

    print(f"找到 {len(valid)} 个 CSV 文件，开始转换：\n")
    converted = 0
    failed = 0
    for c in valid:
        try:
            csv_to_parquet(c)
            converted += 1
        except Exception as e:
            print(f"  {c.name}: 转换失败 - {e}")
            failed += 1

    print(f"\n完成：{converted} 成功, {failed} 失败")
    print(f"Parquet 文件位于: {data_dir}")
    print(f"\n下一步可以跑挖掘：")
    print(f'  .venv\\Scripts\\python.exe -m miaosuan.cli mine --data "{data_dir}\\XAUUSD_H1_5Y.parquet" --market FOREX_XAUUSD --budget deep --out artifacts/mine_xauusd_h1_5y_deep.json --seed 456')


if __name__ == "__main__":
    main()