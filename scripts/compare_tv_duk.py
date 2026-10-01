#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TradingView vs Dukascopy 多源 K 线数据对比验证管线

功能：
1. 从 TradingView 和 Dukascopy 并行拉取指定品种/周期/日期范围的 K 线
2. 按时间戳对齐，逐根比对 OHLCV（价差 ≤ 0.01%，量差 ≤ 1%）
3. 输出详细对比报告：一致率、不一致明细、缺失周期统计
4. 验证通过的数据标准化存为 CSV/Parquet 到 D:/K线数据/{symbol}/{timeframe}.*
5. 记录数据根目录供后续调用

前置条件：
- 需要可访问外网的 HTTP/HTTPS 代理（TradingView、Dukascopy 均在境外）
- 代理可通过 --proxy 参数、环境变量 HTTP_PROXY/HTTPS_PROXY、或系统代理自动探测
- 本地已安装 tvdatafeed (pip install git+https://github.com/rongardF/tvdatafeed.git)

依赖：项目 venv 已安装 tvdatafeed, pandas, pyarrow
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

# 确保能 import miaosuan 模块
sys.path.insert(0, str(Path(__file__).parent.parent))

from miaosuan.data.fetchers.tradingview import TradingViewFetcher
from miaosuan.data.fetchers.dukascopy import DukascopyFetcher
from miaosuan.errors import DataError

# ──────────────────────────────────────────────────────────────────────────────
# 配置常量
# ──────────────────────────────────────────────────────────────────────────────

DATA_ROOT = Path(r"D:/K线数据")
DATA_ROOT.mkdir(parents=True, exist_ok=True)

# 支持的周期（两源都有）
TIMEFRAMES = ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1"]

# 容差设置
PRICE_TOLERANCE_PCT = 0.0001      # 0.01% 价差容差
VOLUME_TOLERANCE_PCT = 0.01       # 1% 量差容差
TIMESTAMP_TOLERANCE_SEC = 60      # 时间戳对齐容差（秒）

# 默认品种映射（TradingView 交易所前缀）
TV_SYMBOL_MAP = {
    "XAUUSD": "TVC:XAUUSD",       # TradingView 现货黄金
    "EURUSD": "FX_IDC:EURUSD",    # 外汇
    "BTCUSDT": "BINANCE:BTCUSDT", # 加密货币（TV 上用 BINANCE 前缀）
    "AUDCAD": "FX_IDC:AUDCAD",
    "GBPUSD": "FX_IDC:GBPUSD",
    "USDJPY": "FX_IDC:USDJPY",
}

# Dukascopy 品种格式已在 fetcher 内部处理 (XAUUSD -> XAU/USD)

# 代理自动探测：环境变量 > 系统代理
def _get_default_proxy() -> str | None:
    """从环境变量获取默认代理"""
    for env_var in ("HTTPS_PROXY", "HTTPS_PROXY", "HTTP_PROXY", "http_proxy", "https_proxy"):
        val = os.environ.get(env_var)
        if val:
            return val
    return None

# ──────────────────────────────────────────────────────────────────────────────
# 数据结构
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BarComparison:
    """单根 K 线对比结果"""
    timestamp: int
    tv_open: float
    duk_open: float
    tv_high: float
    duk_high: float
    tv_low: float
    duk_low: float
    tv_close: float
    duk_close: float
    tv_volume: float
    duk_volume: float
    price_match: bool
    volume_match: bool
    mismatched_fields: list[str]

@dataclass
class TimeframeReport:
    """单周期对比报告"""
    symbol: str
    timeframe: str
    total_bars: int
    matched_bars: int
    price_mismatch: int
    volume_mismatch: int
    timestamp_mismatch: int
    tv_only_bars: int
    duk_only_bars: int
    mismatches: list[dict[str, Any]]  # 详细不一致记录
    start_time: int
    end_time: int
    fetch_duration_tv_sec: float
    fetch_duration_duk_sec: float

@dataclass
class SymbolReport:
    """品种级汇总报告"""
    symbol: str
    timeframes: dict[str, TimeframeReport]
    overall_match_rate: float
    total_bars: int
    total_matched: int
    generated_at: str

# ──────────────────────────────────────────────────────────────────────────────
# 核心工具函数
# ──────────────────────────────────────────────────────────────────────────────

def normalize_symbol_for_tv(symbol: str) -> str:
    """统一品种名到 TradingView 格式"""
    return TV_SYMBOL_MAP.get(symbol.upper(), symbol.upper())

def normalize_symbol_for_duk(symbol: str) -> str:
    """统一品种名到 Dukascopy 格式（fetcher 内部已处理，这里透传）"""
    return symbol.upper()

def fetch_with_timing(fetcher, symbol: str, timeframe: str) -> tuple[pd.DataFrame, float]:
    """带计时的 fetch_full"""
    start = time.perf_counter()
    try:
        df = fetcher.fetch_full(symbol, timeframe)
        duration = time.perf_counter() - start
        return df, duration
    except Exception as e:
        duration = time.perf_counter() - start
        raise DataError(f"{fetcher.source_name} fetch failed after {duration:.1f}s: {e}") from e

def align_and_compare(df_tv: pd.DataFrame, df_duk: pd.DataFrame) -> tuple[list[BarComparison], dict]:
    """
    两源 DataFrame 按 time 对齐比对
    返回：(比对详情列表, 统计摘要)
    """
    # 确保 time 为 int 秒级
    for df in (df_tv, df_duk):
        if "time" not in df.columns:
            raise ValueError("DataFrame 缺少 time 列")
        df["time"] = df["time"].astype("int64")

    # 去重：保留首次出现
    df_tv = df_tv.drop_duplicates(subset="time", keep="first").sort_values("time").reset_index(drop=True)
    df_duk = df_duk.drop_duplicates(subset="time", keep="first").sort_values("time").reset_index(drop=True)

    # 合并：outer join on time，容差匹配
    # 先精确匹配
    merged = pd.merge(
        df_tv, df_duk,
        on="time",
        how="outer",
        suffixes=("_tv", "_duk"),
        indicator=True
    )

    comparisons = []
    stats = {
        "total": len(merged),
        "matched": 0,
        "price_mismatch": 0,
        "volume_mismatch": 0,
        "timestamp_mismatch": 0,  # 仅记录 _merge 不是 both 的
        "tv_only": 0,
        "duk_only": 0,
    }

    for _, row in merged.iterrows():
        ts = int(row["time"])
        tv_present = row.get("open_tv") is not None and pd.notna(row.get("open_tv"))
        duk_present = row.get("open_duk") is not None and pd.notna(row.get("open_duk"))

        if not tv_present:
            stats["duk_only"] += 1
            continue
        if not duk_present:
            stats["tv_only"] += 1
            continue

        # 两源都有，逐字段比对
        mismatched = []
        price_match = True
        volume_match = True

        for field in ["open", "high", "low", "close"]:
            tv_val = row[f"{field}_tv"]
            duk_val = row[f"{field}_duk"]
            if tv_val == 0 and duk_val == 0:
                continue
            if tv_val == 0 or duk_val == 0:
                price_match = False
                mismatched.append(field)
                continue
            pct_diff = abs(tv_val - duk_val) / max(abs(tv_val), abs(duk_val))
            if pct_diff > PRICE_TOLERANCE_PCT:
                price_match = False
                mismatched.append(field)

        # volume 比对
        tv_vol = row.get("volume_tv", 0.0)
        duk_vol = row.get("volume_duk", 0.0)
        if tv_vol == 0 and duk_vol == 0:
            pass
        elif tv_vol == 0 or duk_vol == 0:
            volume_match = False
            mismatched.append("volume")
        else:
            pct_diff = abs(tv_vol - duk_vol) / max(abs(tv_vol), abs(duk_vol))
            if pct_diff > VOLUME_TOLERANCE_PCT:
                volume_match = False
                mismatched.append("volume")

        comp = BarComparison(
            timestamp=ts,
            tv_open=row["open_tv"], duk_open=row["open_duk"],
            tv_high=row["high_tv"], duk_high=row["high_duk"],
            tv_low=row["low_tv"], duk_low=row["low_duk"],
            tv_close=row["close_tv"], duk_close=row["close_duk"],
            tv_volume=tv_vol, duk_volume=duk_vol,
            price_match=price_match,
            volume_match=volume_match,
            mismatched_fields=mismatched,
        )
        comparisons.append(comp)

        if price_match and volume_match:
            stats["matched"] += 1
        else:
            if not price_match:
                stats["price_mismatch"] += 1
            if not volume_match:
                stats["volume_mismatch"] += 1

    return comparisons, stats

def save_standardized_data(df: pd.DataFrame, symbol: str, timeframe: str) -> tuple[Path, Path]:
    """
    保存标准化数据到 D:/K线数据/{symbol}/{timeframe}.csv/.parquet
    标准列：time, open, high, low, close, volume
    """
    sym_dir = DATA_ROOT / symbol.upper()
    sym_dir.mkdir(parents=True, exist_ok=True)

    # 标准化列顺序与类型
    out_df = df[["time", "open", "high", "low", "close", "volume"]].copy()
    out_df["time"] = out_df["time"].astype("int64")
    for col in ["open", "high", "low", "close", "volume"]:
        out_df[col] = out_df[col].astype("float64")
    out_df = out_df.sort_values("time").reset_index(drop=True)

    csv_path = sym_dir / f"{timeframe}.csv"
    pq_path = sym_dir / f"{timeframe}.parquet"

    out_df.to_csv(csv_path, index=False)
    out_df.to_parquet(pq_path, index=False, engine="pyarrow")

    return csv_path, pq_path

def load_existing_standardized(symbol: str, timeframe: str) -> pd.DataFrame | None:
    """加载已存在的标准化数据（优先 parquet），搜索标准目录和 cache 目录"""
    # 1. 标准目录
    sym_dir = DATA_ROOT / symbol.upper()
    pq_path = sym_dir / f"{timeframe}.parquet"
    csv_path = sym_dir / f"{timeframe}.csv"
    if pq_path.exists():
        return pd.read_parquet(pq_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)

    # 2. cache 目录（NetworkSource 落盘位置）
    cache_dir = Path(__file__).parent.parent / "data" / "cache"
    cache_pq = cache_dir / f"{symbol.upper()}_{timeframe}_network.parquet"
    if cache_pq.exists():
        df = pd.read_parquet(cache_pq)
        # 确保标准列
        if "time" in df.columns:
            return df[["time", "open", "high", "low", "close", "volume"]].copy()
    return None

# ──────────────────────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────────────────────

def compare_symbol_timeframe(
    symbol: str,
    timeframe: str,
    tv_fetcher: TradingViewFetcher,
    duk_fetcher: DukascopyFetcher,
    start_date: str | None = None,
    end_date: str | None = None,
) -> TimeframeReport:
    """对比单品种单周期"""
    tv_sym = normalize_symbol_for_tv(symbol)
    duk_sym = normalize_symbol_for_duk(symbol)

    print(f"  [{symbol} {timeframe}] 拉取 TradingView ({tv_sym})...", end=" ", flush=True)
    df_tv, dur_tv = fetch_with_timing(tv_fetcher, tv_sym, timeframe)
    print(f"{len(df_tv)} 根, {dur_tv:.1f}s")

    print(f"  [{symbol} {timeframe}] 拉取 Dukascopy ({duk_sym})...", end=" ", flush=True)
    df_duk, dur_duk = fetch_with_timing(duk_fetcher, duk_sym, timeframe)
    print(f"{len(df_duk)} 根, {dur_duk:.1f}s")

    # 日期过滤
    if start_date:
        start_ts = int(datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc).timestamp())
        df_tv = df_tv[df_tv["time"] >= start_ts]
        df_duk = df_duk[df_duk["time"] >= start_ts]
    if end_date:
        end_ts = int(datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc).timestamp())
        df_tv = df_tv[df_tv["time"] <= end_ts]
        df_duk = df_duk[df_duk["time"] <= end_ts]

    # 对齐比对
    comparisons, stats = align_and_compare(df_tv, df_duk)

    # 构建不一致明细（仅保留前 100 条避免报告过大）
    mismatches = []
    for c in comparisons:
        if not (c.price_match and c.volume_match):
            mismatches.append({
                "timestamp": c.timestamp,
                "datetime_utc": datetime.fromtimestamp(c.timestamp, tz=timezone.utc).isoformat(),
                "mismatched_fields": c.mismatched_fields,
                "tv": {k: getattr(c, f"tv_{k}") for k in ["open","high","low","close","volume"]},
                "duk": {k: getattr(c, f"duk_{k}") for k in ["open","high","low","close","volume"]},
            })
    mismatches = mismatches[:100]  # 截断

    report = TimeframeReport(
        symbol=symbol.upper(),
        timeframe=timeframe,
        total_bars=stats["total"],
        matched_bars=stats["matched"],
        price_mismatch=stats["price_mismatch"],
        volume_mismatch=stats["volume_mismatch"],
        timestamp_mismatch=stats["tv_only"] + stats["duk_only"],
        tv_only_bars=stats["tv_only"],
        duk_only_bars=stats["duk_only"],
        mismatches=mismatches,
        start_time=int(df_tv["time"].min()) if not df_tv.empty else 0,
        end_time=int(df_tv["time"].max()) if not df_tv.empty else 0,
        fetch_duration_tv_sec=dur_tv,
        fetch_duration_duk_sec=dur_duk,
    )

    match_rate = stats["matched"] / stats["total"] * 100 if stats["total"] > 0 else 0
    print(f"  [{symbol} {timeframe}] 一致率: {match_rate:.2f}% ({stats['matched']}/{stats['total']})")

    # 若一致率 ≥ 95%，保存标准化数据（取两源交集的 matched 部分）
    if match_rate >= 95.0 and stats["matched"] > 0:
        # 取 matched 的时间戳
        matched_ts = {c.timestamp for c in comparisons if c.price_match and c.volume_match}
        # 合并两源数据（优先 TV，缺失用 Dukascopy）
        df_combined = pd.concat([df_tv, df_duk]).drop_duplicates(subset="time", keep="first")
        df_matched = df_combined[df_combined["time"].isin(matched_ts)].copy()
        if not df_matched.empty:
            csv_p, pq_p = save_standardized_data(df_matched, symbol, timeframe)
            print(f"    → 已保存标准化数据: {csv_p}, {pq_p}")

    return report

def run_comparison(
    symbols: list[str],
    timeframes: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    proxy: str | None = None,
) -> dict[str, SymbolReport]:
    """运行全量对比"""
    tf_list = timeframes or TIMEFRAMES

    # 代理优先级：显式参数 > 环境变量 > 系统代理（fetcher 内部探测）
    effective_proxy = proxy or _get_default_proxy()
    if effective_proxy:
        print(f"🔗 使用代理: {effective_proxy}")
    else:
        print("⚠ 未检测到代理，TradingView/Dukascopy 可能无法连通（需配置 HTTP_PROXY 或 --proxy）")

    # 初始化 fetcher
    tv_fetcher = TradingViewFetcher(proxy=effective_proxy, max_bars=10000)
    duk_fetcher = DukascopyFetcher(max_bars=10000, window_days=365*10)  # 10 年窗口

    # 可用性检查
    tv_available = tv_fetcher.is_available()
    duk_available = duk_fetcher.is_available()

    if not tv_available:
        print("⚠ TradingView 不可用（tvdatafeed 未安装或网络不通）")
    if not duk_available:
        print("⚠ Dukascopy 不可用（主机不可达）")

    if not tv_available and not duk_available:
        print("❌ 两源均不可用，尝试加载本地已有标准化数据进行自检...")
        return run_local_only_check(symbols, tf_list, start_date, end_date)

    all_reports = {}

    for symbol in symbols:
        print(f"\n=== 开始对比 {symbol} ===")
        tf_reports = {}

        for tf in tf_list:
            try:
                report = compare_symbol_timeframe(symbol, tf, tv_fetcher, duk_fetcher, start_date, end_date)
                tf_reports[tf] = report
            except DataError as e:
                print(f"  [{symbol} {tf}] 数据错误: {e}")
                tf_reports[tf] = TimeframeReport(
                    symbol=symbol.upper(), timeframe=tf,
                    total_bars=0, matched_bars=0, price_mismatch=0,
                    volume_mismatch=0, timestamp_mismatch=0,
                    tv_only_bars=0, duk_only_bars=0,
                    mismatches=[], start_time=0, end_time=0,
                    fetch_duration_tv_sec=0, fetch_duration_duk_sec=0,
                )
            except Exception as e:
                print(f"  [{symbol} {tf}] 意外错误: {e}")
                tf_reports[tf] = TimeframeReport(
                    symbol=symbol.upper(), timeframe=tf,
                    total_bars=0, matched_bars=0, price_mismatch=0,
                    volume_mismatch=0, timestamp_mismatch=0,
                    tv_only_bars=0, duk_only_bars=0,
                    mismatches=[{"error": str(e)}], start_time=0, end_time=0,
                    fetch_duration_tv_sec=0, fetch_duration_duk_sec=0,
                )

        # 品种级汇总
        total_bars = sum(r.total_bars for r in tf_reports.values())
        total_matched = sum(r.matched_bars for r in tf_reports.values())
        overall_rate = total_matched / total_bars * 100 if total_bars > 0 else 0

        all_reports[symbol.upper()] = SymbolReport(
            symbol=symbol.upper(),
            timeframes=tf_reports,
            overall_match_rate=overall_rate,
            total_bars=total_bars,
            total_matched=total_matched,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )
        print(f"  {symbol} 总体一致率: {overall_rate:.2f}% ({total_matched}/{total_bars})")

    return all_reports


def run_local_only_check(
    symbols: list[str],
    timeframes: list[str],
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, SymbolReport]:
    """网络源均不可用时：加载本地标准化数据做自检（完整性、异常值检测），并落盘到标准目录"""
    print("📂 进入本地数据自检模式...")
    all_reports = {}

    for symbol in symbols:
        print(f"\n=== 本地自检 {symbol} ===")
        tf_reports = {}

        for tf in timeframes:
            df = load_existing_standardized(symbol, tf)
            if df is None or df.empty:
                print(f"  [{symbol} {tf}] 无本地数据")
                tf_reports[tf] = TimeframeReport(
                    symbol=symbol.upper(), timeframe=tf,
                    total_bars=0, matched_bars=0, price_mismatch=0,
                    volume_mismatch=0, timestamp_mismatch=0,
                    tv_only_bars=0, duk_only_bars=0,
                    mismatches=[{"warning": "无本地数据"}], start_time=0, end_time=0,
                    fetch_duration_tv_sec=0, fetch_duration_duk_sec=0,
                )
                continue

            # 日期过滤
            if start_date:
                start_ts = int(datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc).timestamp())
                df = df[df["time"] >= start_ts]
            if end_date:
                end_ts = int(datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc).timestamp())
                df = df[df["time"] <= end_ts]

            # 自检：连续性、异常值、零成交量
            mismatches = []
            df = df.sort_values("time").reset_index(drop=True)

            # 时间间隙检测
            expected_intervals = {"M1": 60, "M5": 300, "M15": 900, "M30": 1800,
                                  "H1": 3600, "H4": 14400, "D1": 86400, "W1": 604800}
            expected_sec = expected_intervals.get(tf, 3600)
            gaps = df["time"].diff().dropna()
            gap_count = int((gaps > expected_sec * 1.5).sum())

            # 异常值检测
            for col in ["open", "high", "low", "close"]:
                q1, q3 = df[col].quantile([0.25, 0.75])
                iqr = q3 - q1
                outliers = df[(df[col] < q1 - 3*iqr) | (df[col] > q3 + 3*iqr)]
                if len(outliers) > 0:
                    mismatches.append({
                        "type": "price_outlier",
                        "field": col,
                        "count": int(len(outliers)),
                        "detail": outliers[["time", col]].head(5).to_dict("records"),
                    })

            # 零成交量
            zero_vol = int((df["volume"] == 0).sum())
            if zero_vol > 0:
                mismatches.append({
                    "type": "zero_volume",
                    "count": zero_vol,
                })

            tf_reports[tf] = TimeframeReport(
                symbol=symbol.upper(), timeframe=tf,
                total_bars=len(df), matched_bars=len(df) - len(mismatches),
                price_mismatch=0, volume_mismatch=0, timestamp_mismatch=gap_count,
                tv_only_bars=0, duk_only_bars=0,
                mismatches=mismatches,
                start_time=int(df["time"].min()) if not df.empty else 0,
                end_time=int(df["time"].max()) if not df.empty else 0,
                fetch_duration_tv_sec=0, fetch_duration_duk_sec=0,
            )
            print(f"  [{symbol} {tf}] 本地 {len(df)} 根，间隙 {gap_count}，异常 {len(mismatches)}")

            # 保存标准化数据到标准目录
            if len(df) > 0:
                csv_p, pq_p = save_standardized_data(df, symbol, tf)
                print(f"    → 已保存标准化数据: {csv_p}, {pq_p}")

        total_bars = sum(r.total_bars for r in tf_reports.values())
        total_matched = sum(r.matched_bars for r in tf_reports.values())
        overall_rate = total_matched / total_bars * 100 if total_bars > 0 else 0

        all_reports[symbol.upper()] = SymbolReport(
            symbol=symbol.upper(),
            timeframes=tf_reports,
            overall_match_rate=overall_rate,
            total_bars=total_bars,
            total_matched=total_matched,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

    return all_reports

def generate_report(all_reports: dict[str, SymbolReport], output_path: Path) -> None:
    """生成 JSON 报告"""
    report_data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "price_tolerance_pct": PRICE_TOLERANCE_PCT * 100,
            "volume_tolerance_pct": VOLUME_TOLERANCE_PCT * 100,
            "timestamp_tolerance_sec": TIMESTAMP_TOLERANCE_SEC,
            "data_root": str(DATA_ROOT),
        },
        "symbols": {},
    }

    for sym, rep in all_reports.items():
        report_data["symbols"][sym] = {
            "overall_match_rate": rep.overall_match_rate,
            "total_bars": rep.total_bars,
            "total_matched": rep.total_matched,
            "timeframes": {},
        }
        for tf, tf_rep in rep.timeframes.items():
            report_data["symbols"][sym]["timeframes"][tf] = {
                "total_bars": tf_rep.total_bars,
                "matched_bars": tf_rep.matched_bars,
                "match_rate_pct": (tf_rep.matched_bars / tf_rep.total_bars * 100) if tf_rep.total_bars > 0 else 0,
                "price_mismatch": tf_rep.price_mismatch,
                "volume_mismatch": tf_rep.volume_mismatch,
                "timestamp_mismatch": tf_rep.timestamp_mismatch,
                "tv_only_bars": tf_rep.tv_only_bars,
                "duk_only_bars": tf_rep.duk_only_bars,
                "mismatch_details": tf_rep.mismatches,
                "time_range": {
                    "start": tf_rep.start_time,
                    "end": tf_rep.end_time,
                    "start_utc": datetime.fromtimestamp(tf_rep.start_time, tz=timezone.utc).isoformat() if tf_rep.start_time else None,
                    "end_utc": datetime.fromtimestamp(tf_rep.end_time, tz=timezone.utc).isoformat() if tf_rep.end_time else None,
                },
                "fetch_duration_sec": {
                    "tradingview": tf_rep.fetch_duration_tv_sec,
                    "dukascopy": tf_rep.fetch_duration_duk_sec,
                },
            }

    output_path.write_text(json.dumps(report_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n📊 对比报告已保存: {output_path}")

def print_summary(all_reports: dict[str, SymbolReport]) -> None:
    """控制台打印汇总"""
    print("\n" + "="*70)
    print("对比验证汇总")
    print("="*70)
    for sym, rep in all_reports.items():
        print(f"\n品种: {sym}")
        print(f"  总体一致率: {rep.overall_match_rate:.2f}% ({rep.total_matched}/{rep.total_bars})")
        for tf, tf_rep in rep.timeframes.items():
            rate = (tf_rep.matched_bars / tf_rep.total_bars * 100) if tf_rep.total_bars > 0 else 0
            status = "✅" if rate >= 95 else "⚠️" if rate >= 80 else "❌"
            print(f"  {status} {tf:>4s}: {rate:6.2f}% ({tf_rep.matched_bars:>5}/{tf_rep.total_bars:>5}) "
                  f"价差:{tf_rep.price_mismatch} 量差:{tf_rep.volume_mismatch} 仅TV:{tf_rep.tv_only_bars} 仅Duk:{tf_rep.duk_only_bars}")
    print(f"\n📁 标准化数据根目录: {DATA_ROOT}")
    print("   结构: {symbol}/{timeframe}.csv|.parquet")

# ──────────────────────────────────────────────────────────────────────────────
# CLI 入口
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TradingView vs Dukascopy K线数据对比验证")
    parser.add_argument("symbols", nargs="+", help="交易品种（如 XAUUSD EURUSD BTCUSDT）")
    parser.add_argument("--timeframes", "-t", nargs="+", default=TIMEFRAMES,
                        help=f"周期列表（默认全周期: {' '.join(TIMEFRAMES)}）")
    parser.add_argument("--start", "-s", help="开始日期 (YYYY-MM-DD)")
    parser.add_argument("--end", "-e", help="结束日期 (YYYY-MM-DD)")
    parser.add_argument("--proxy", "-p", help="HTTP 代理 (如 http://127.0.0.1:7890)，优先于环境变量")
    parser.add_argument("--output", "-o", default=DATA_ROOT / "comparison_report.json",
                        help="报告输出路径")
    parser.add_argument("--local-only", action="store_true",
                        help="仅做本地数据自检（不联网），用于离线环境")
    args = parser.parse_args()

    print(f"🚀 启动对比验证: {', '.join(args.symbols)}")
    print(f"   周期: {', '.join(args.timeframes)}")
    print(f"   日期范围: {args.start or '最早'} ~ {args.end or '最新'}")
    print(f"   价差容差: {PRICE_TOLERANCE_PCT*100:.4f}%, 量差容差: {VOLUME_TOLERANCE_PCT*100:.1f}%")
    print(f"   数据根目录: {DATA_ROOT}")
    if args.local_only:
        print("   模式: 本地自检（不联网）")

    if args.local_only:
        reports = run_local_only_check(
            symbols=args.symbols,
            timeframes=args.timeframes,
            start_date=args.start,
            end_date=args.end,
        )
    else:
        reports = run_comparison(
            symbols=args.symbols,
            timeframes=args.timeframes,
            start_date=args.start,
            end_date=args.end,
            proxy=args.proxy,
        )

    generate_report(reports, Path(args.output))
    print_summary(reports)

    # 记录数据根目录配置（供后续调用）
    config_path = DATA_ROOT / ".data_root.json"
    config_path.write_text(json.dumps({
        "data_root": str(DATA_ROOT),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "symbols": list(reports.keys()),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"🔖 数据根目录配置已记录: {config_path}")

if __name__ == "__main__":
    main()