# -*- coding: utf-8 -*-
# Copyright (c) 2026 Original Author
# Copyright (c) 2026 Your Company
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""MT4/外汇 Parquet 数据源（MVP 唯一实现，架构 §2 ``data/sources/parquet_mt.py``）。

封装 :mod:`miaosuan.data.loader` 的 Parquet 语义（成交量列名 ``tick_volume`` 优先、
``time<1e7 → ×1000``、排序去重、``float32``），并给出**复权口径**与**文件名契约**
``{symbol}_{timeframe}.parquet``。

归属说明（架构 §9.4）：**复权 / 日历归本层**，``MarketProfile`` 不感知数据来源。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .. import loader
from ..panel import Panel
from .base import BaseDataSource

__all__ = ["ParquetMTDataSource"]


class ParquetMTDataSource(BaseDataSource):
    """MT4/MT5 导出的 Parquet 行情源。

    :param path: ``{symbol}_{timeframe}.parquet`` 文件路径。
    :param symbols: 品种名（``None`` → 从文件名推断）。
    :param timeframe: 周期（``None`` → 从文件名推断）。
    :param volume_column: 显式成交量列（``None`` → ``tick_volume`` 优先，否则 ``volume``）。
    :param adjustment_mode: 复权口径标签（**仅记录**；外汇/贵金属现货通常 ``"none"``）。
    :param market_profile_name: 评估所用市场画像名（写入 ``Panel``，可追溯）。
    """

    source_id = "parquet_mt"

    def __init__(
        self,
        path: str | Path,
        *,
        symbols: tuple[str, ...] | None = None,
        timeframe: str | None = None,
        volume_column: str | None = None,
        adjustment_mode: str = "none",
        market_profile_name: str = "",
    ) -> None:
        self.path = Path(path)
        inferred_symbol, inferred_tf = loader.infer_symbol_timeframe(self.path)
        resolved_symbols = tuple(symbols) if symbols else (inferred_symbol,)
        resolved_tf = timeframe if timeframe is not None else inferred_tf
        super().__init__(resolved_symbols, resolved_tf, adjustment_mode=adjustment_mode)
        self.volume_column = volume_column
        self.market_profile_name = market_profile_name

    def load(self, spec: dict[str, Any] | None = None) -> Panel:
        """加载 Parquet → :class:`Panel`（``spec`` 覆盖构造默认值）。"""
        overrides: dict[str, Any] = spec or {}
        path = Path(overrides.get("path", self.path))
        panel = loader.load_parquet(
            path,
            symbols=tuple(overrides.get("symbols", self.symbols)),
            timeframe=str(overrides.get("timeframe", self.timeframe)),
            market_profile_name=str(
                overrides.get("market_profile_name", self.market_profile_name)
            ),
            volume_column=overrides.get("volume_column", self.volume_column),
            time_unit=str(overrides.get("time_unit", "auto")),
            adjustment_mode=str(overrides.get("adjustment_mode", self.adjustment_mode())),
            meta={"source_id": self.source_id},
        )
        self._panel = panel
        return panel
