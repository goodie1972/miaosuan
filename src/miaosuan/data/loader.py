# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""数据加载器：Parquet / CSV → :class:`~miaosuan.data.panel.Panel`（架构 §2）。

复刻冻结 AlphaMaster ``data_pipeline`` 的数据语义（详见
``docs/data_adaptation_checklist.md``），并补齐 CSV 通道：

* **成交量列名**：``tick_volume``（MT5 导出）优先，否则 ``volume``；
* **时间戳单位修复**：``time.max() < 1e7`` 视为「秒/1000」→ 乘 1000（同时支持显式
  ``time_unit="s"/"ms"``）；
* **排序 + 去重**：按 ``time`` 升序，重复时间戳保留**最后一条**（``keep="last"``）；
* **dtype**：OHLCV 统一 ``float32``，形状 ``[1, T]``（单品种）；
* **缺失值**：统一 ``np.nan``（**禁止**用 0 填充价格）。

本模块属数据层，允许文件 IO 与 pandas；（``core/`` 不 import 本模块）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..errors import DataError
from .panel import PANEL_FIELDS, Panel

__all__ = [
    "TIME_SCALE_THRESHOLD",
    "infer_symbol_timeframe",
    "load",
    "load_csv",
    "load_parquet",
    "panel_from_frame",
]

#: 时间戳「秒/1000」判据（1970-04-27 之前），与 AM ``parquet_manager.py`` 一致
TIME_SCALE_THRESHOLD: int = 10_000_000

#: 周期别名 → 规范名（覆盖常见写法；未知则原样大写）
_TIMEFRAME_ALIASES: dict[str, str] = {
    "m1": "M1", "1m": "M1", "1min": "M1",
    "m5": "M5", "5m": "M5", "5min": "M5",
    "m15": "M15", "15m": "M15", "15min": "M15",
    "m30": "M30", "30m": "M30", "30min": "M30",
    "h1": "H1", "1h": "H1", "60m": "H1", "60min": "H1",
    "h4": "H4", "4h": "H4", "240m": "H4", "240min": "H4",
    "d1": "D1", "1d": "D1", "1day": "D1", "daily": "D1",
    "w1": "W1", "1w": "W1", "weekly": "W1",
    "mn1": "MN1", "1mo": "MN1", "monthly": "MN1",
}

_TIME_UNITS = ("auto", "s", "ms")


def normalize_timeframe(token: str) -> str:
    """把文件名中的周期 token 规范化为 ``M1/M5/.../H1/D1``（未知则原样大写）。"""
    raw = (token or "").strip()
    if not raw:
        return ""
    key = raw.lower().replace("-", "").replace("_", "")
    return _TIMEFRAME_ALIASES.get(key, raw.upper())


def infer_symbol_timeframe(path: str | Path) -> tuple[str, str]:
    """从 ``{symbol}_{timeframe}.parquet`` 推断 ``(symbol, timeframe)``。

    无下划线时，``symbol`` = 文件名主体、``timeframe`` = 空串。
    """
    stem = Path(path).stem
    if "_" in stem:
        symbol, tf_raw = stem.rsplit("_", 1)
        return symbol.strip() or stem, normalize_timeframe(tf_raw)
    return stem, ""


def _normalize_time(series: Any, time_unit: str) -> np.ndarray:
    """把时间列规范化为 int64 Unix 秒（含 AM 的秒/1000 修复）。"""
    if time_unit not in _TIME_UNITS:
        raise DataError(f"time_unit 非法: {time_unit!r}（可选 {_TIME_UNITS}）")
    values = np.asarray(series.to_numpy())
    if not np.issubdtype(values.dtype, np.number):
        raise DataError("time 列必须为数值型（Unix 时间戳）", context={"dtype": str(values.dtype)})
    as_int = values.astype(np.int64)
    if time_unit == "ms":
        return as_int // 1000
    if time_unit == "s":
        return as_int
    # auto：max < 1e7 视为被除过 1000
    if as_int.size and int(as_int.max()) < TIME_SCALE_THRESHOLD:
        return as_int * 1000
    return as_int


def _column_to_panel_field(series: Any) -> np.ndarray:
    """单列 → ``[1, T]`` float32（缺失值保留为 NaN）。"""
    arr = np.asarray(series.to_numpy(), dtype=np.float32)
    return arr.reshape(1, -1)


def _resolve_volume_column(df: Any, volume_column: str | None) -> str:
    if volume_column is not None:
        if volume_column not in df.columns:
            raise DataError(f"指定的成交量列不存在: {volume_column!r}")
        return volume_column
    if "tick_volume" in df.columns:
        return "tick_volume"
    if "volume" in df.columns:
        return "volume"
    raise DataError("缺少成交量列（需 tick_volume 或 volume）")


def panel_from_frame(
    df: Any,
    *,
    symbols: tuple[str, ...] | None = None,
    timeframe: str = "",
    market_profile_name: str = "",
    volume_column: str | None = None,
    time_unit: str = "auto",
    adjustment_mode: str = "none",
    meta: dict[str, Any] | None = None,
) -> Panel:
    """把 OHLCV DataFrame → :class:`Panel`（复刻 AM 语义）。

    :param df: 含 ``time/open/high/low/close/(tick_)volume`` 列的 DataFrame。
    :param symbols: 品种名；``None`` 时对单品种默认 ``("SINGLE",)``。
    :param volume_column: 显式指定成交量列；``None`` 时按 ``tick_volume``→``volume``。
    :param time_unit: ``"auto" | "s" | "ms"``。
    :param adjustment_mode: 复权口径标签（**仅记录在 meta**；复权本身由 DataSource 负责）。
    """
    if df is None or len(df) == 0:
        raise DataError("输入数据为空")

    vcol = _resolve_volume_column(df, volume_column)
    required = ["time", "open", "high", "low", "close", vcol]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataError(f"输入缺少列: {missing}", context={"missing": missing})

    sub = df[required].copy().rename(columns={vcol: "volume"})
    sub["time"] = _normalize_time(sub["time"], time_unit)
    sub = sub.sort_values("time")
    sub = sub[~sub["time"].duplicated(keep="last")]

    fields = {name: _column_to_panel_field(sub[name]) for name in PANEL_FIELDS}
    time = np.asarray(sub["time"].to_numpy(), dtype=np.int64)

    n = int(fields["open"].shape[0])
    if symbols is None:
        symbols = ("SINGLE",) if n == 1 else tuple(f"S{i}" for i in range(n))
    if len(symbols) != n:
        raise DataError(
            "symbols 长度与数据行数不一致",
            context={"len_symbols": len(symbols), "rows": n},
        )

    resolved_meta: dict[str, Any] = {
        "adjustment_mode": adjustment_mode,
        "volume_column": vcol,
        "time_unit": time_unit,
        "rows_raw": int(len(df)),
        "rows_used": int(time.shape[0]),
    }
    if meta:
        resolved_meta.update(meta)
    return Panel.from_arrays(
        fields,
        time,
        symbols=tuple(symbols),
        timeframe=timeframe,
        market_profile_name=market_profile_name,
        meta=resolved_meta,
    )


def load_parquet(path: str | Path, **kwargs: Any) -> Panel:
    """加载 Parquet（``**kwargs`` 透传 :func:`panel_from_frame`）。"""
    p = Path(path)
    if not p.is_file():
        raise DataError(f"Parquet 文件不存在: {p}")
    df = pd.read_parquet(p)
    kwargs["meta"] = _with_source_meta(kwargs.get("meta"), str(p), "parquet")
    return panel_from_frame(df, **kwargs)


def load_csv(path: str | Path, **kwargs: Any) -> Panel:
    """加载 CSV（``**kwargs`` 透传 :func:`panel_from_frame`）。"""
    p = Path(path)
    if not p.is_file():
        raise DataError(f"CSV 文件不存在: {p}")
    df = pd.read_csv(p)
    kwargs["meta"] = _with_source_meta(kwargs.get("meta"), str(p), "csv")
    return panel_from_frame(df, **kwargs)


def _with_source_meta(meta: dict[str, Any] | None, source_path: str, fmt: str) -> dict[str, Any]:
    """复制并补充来源元信息（不修改调用方传入的 dict）。"""
    resolved = dict(meta) if meta else {}
    resolved.setdefault("source_path", source_path)
    resolved.setdefault("format", fmt)
    return resolved


def load(path: str | Path, **kwargs: Any) -> Panel:
    """按扩展名分派到 :func:`load_parquet` / :func:`load_csv`。

    未显式给 ``symbols`` / ``timeframe`` 时，从文件名 ``{symbol}_{timeframe}`` 推断。
    """
    p = Path(path)
    suffix = p.suffix.lower()
    symbol, timeframe = infer_symbol_timeframe(p)
    kwargs.setdefault("symbols", (symbol,))
    kwargs.setdefault("timeframe", timeframe)
    if suffix in (".parquet", ".pq"):
        return load_parquet(p, **kwargs)
    if suffix in (".csv", ".txt"):
        return load_csv(p, **kwargs)
    raise DataError(f"不支持的数据格式: {suffix!r}（支持 .parquet/.csv）", context={"path": str(p)})
