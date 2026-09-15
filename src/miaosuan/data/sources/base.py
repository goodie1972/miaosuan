# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""``DataSource`` 实现基类（架构 §2「每产品一个源」）。

本模块落地 :class:`~miaosuan.core.ports.DataSource` 协议的**公共骨架**：源标识、品种/周期、
复权口径、交易日历与可交易掩码。**复权 / 日历 / 缺失处理 / 时间索引**归此层
（架构 §9.4 硬分界：这些**不属于** ``MarketProfile``）。

实现者只需重写 :meth:`load`（把规格 → :class:`~miaosuan.data.panel.Panel`）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ..errors import DataError
from ..panel import Panel

__all__ = ["BaseDataSource"]


class BaseDataSource(ABC):
    """:class:`~miaosuan.core.ports.DataSource` 协议的公共实现骨架。

    结构性实现 ``core/ports.py`` 的 ``DataSource`` 协议（``source_id`` / ``symbols`` /
    ``timeframe`` + ``load`` / ``adjustment_mode`` / ``trading_calendar`` / ``tradable_mask``）。

    :param symbols: 品种名序列。
    :param timeframe: 周期标识（如 ``"H1"``）。
    :param adjustment_mode: 复权口径（``"none"`` / ``"forward"`` / ``"backward"``）。
    """

    #: 源标识（子类覆盖；写入 ``Panel.meta``）
    source_id: str = "base"

    def __init__(
        self,
        symbols: tuple[str, ...] | list[str],
        timeframe: str,
        *,
        adjustment_mode: str = "none",
    ) -> None:
        self.symbols: list[str] = [str(s) for s in symbols]
        self.timeframe: str = str(timeframe)
        self._adjustment_mode: str = str(adjustment_mode)
        self._panel: Panel | None = None

    # ── 协议：加载 ──────────────────────────────────────────────────────────

    @abstractmethod
    def load(self, spec: dict[str, Any] | None = None) -> Panel:
        """按规格加载数据，返回 :class:`Panel`（并缓存为 ``self._panel``）。

        :param spec: 规格覆盖项（文件路径 / 品种 / 周期 / 复权模式等）；``None`` 用构造默认。
        """

    # ── 协议：复权 / 日历 / 掩码 ─────────────────────────────────────────────

    def adjustment_mode(self) -> str:
        """返回复权模式（复权归 ``DataSource``，不进 ``MarketProfile``）。"""
        return self._adjustment_mode

    def trading_calendar(self) -> np.ndarray:
        """返回交易日历（已加载 Panel 的 ``time``，UTC 秒）。"""
        return self._require_panel().time.copy()

    def tradable_mask(self) -> np.ndarray:
        """返回逐 bar 可交易掩码 ``[T]`` bool。

        优先取 ``Panel.meta["tradable_mask"]``（由源在加载时生成，如停牌/无报价）；
        未提供时默认全部可交易。
        """
        panel = self._require_panel()
        raw = panel.meta.get("tradable_mask")
        if raw is None:
            return np.ones(panel.n_bars, dtype=bool)
        mask = np.asarray(raw, dtype=bool)
        if mask.shape != (panel.n_bars,):
            raise DataError(
                "tradable_mask 形状必须为 [T]",
                context={"mask_shape": mask.shape, "T": panel.n_bars},
            )
        return mask

    # ── 便捷 ────────────────────────────────────────────────────────────────

    @property
    def panel(self) -> Panel:
        """已加载的 :class:`Panel`（未加载则报错）。"""
        return self._require_panel()

    def _require_panel(self) -> Panel:
        if self._panel is None:
            raise DataError(f"[{self.source_id}] 尚未加载数据：请先调用 load()")
        return self._panel
