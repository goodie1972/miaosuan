# -*- coding: utf-8 -*-
# Copyright (c) 2026 Original Author
# Copyright (c) 2026 Your Company
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""``Panel`` —— OHLCV 数据契约（架构 §2 / §9.4）。

``Panel`` 是**产品轴与数据源的纯数据容器**：只有 OHLCV + time + symbols + fingerprint
+ ``market_profile_name``，**不含任何市场知识**（成本 / 多空 / 杠杆 / 时段 / 结算一律
归 :class:`~miaosuan.core.ports.MarketProfile`；复权 / 日历 / 缺失处理归 ``DataSource``）。

统一契约（架构 §9.4）：

* 张量形状 ``[N_symbols, T_bars]``，dtype ``float32``；
* 时间统一 **UTC epoch 秒（int64）**；
* 缺失值统一 ``np.nan``，**禁止**用 0 填充价格；
* 自带 ``market_profile_name``，使产物可追溯「这份数据按哪个市场的规则评估」；
* ``fingerprint`` = sha256(OHLCV + time + symbols)。

本模块是纯数据容器，无 IO、无环境变量；文件读写一律在 :mod:`miaosuan.data.loader`。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from ..errors import DataError
from .fingerprint import panel_fingerprint

__all__ = ["Panel", "PANEL_FIELDS"]

#: OHLCV 字段的规范顺序（跨模块统一）
PANEL_FIELDS: tuple[str, ...] = ("open", "high", "low", "close", "volume")


@dataclass(eq=False, repr=False)
class Panel:
    """行情面板（纯数据容器）。

    :param open: ``[N, T]`` float32 开盘价。
    :param high: ``[N, T]`` float32 最高价。
    :param low: ``[N, T]`` float32 最低价。
    :param close: ``[N, T]`` float32 收盘价。
    :param volume: ``[N, T]`` float32 成交量。
    :param time: ``[T]`` int64 Unix 秒（严格递增、唯一）。
    :param symbols: 品种名（长度 == N，顺序与第一维对齐）。
    :param timeframe: 周期标识（如 ``"H1"``）。
    :param market_profile_name: 评估所用市场画像名（可追溯；空串表示未指定）。
    :param meta: 数据来源等附加信息（不参与指纹）。
    """

    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    time: np.ndarray
    symbols: tuple[str, ...]
    timeframe: str = ""
    market_profile_name: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._validate()

    # ── 校验 ────────────────────────────────────────────────────────────────

    def _validate(self) -> None:
        shapes: dict[str, tuple[int, ...]] = {}
        for name in PANEL_FIELDS:
            arr = getattr(self, name)
            if not isinstance(arr, np.ndarray):
                raise DataError(
                    f"Panel.{name} 必须是 numpy 数组", context={"field": name, "type": type(arr)}
                )
            if arr.ndim != 2:
                raise DataError(
                    f"Panel.{name} 必须为 2 维 [N, T]", context={"field": name, "shape": arr.shape}
                )
            shapes[name] = arr.shape

        unique_shapes = set(shapes.values())
        if len(unique_shapes) != 1:
            raise DataError("Panel 各字段形状必须一致 [N, T]", context={"shapes": shapes})
        n, t = shapes["open"]

        if self.time.ndim != 1 or self.time.shape[0] != t:
            raise DataError(
                "Panel.time 必须为 1 维且长度等于 T",
                context={"time_shape": self.time.shape, "T": t},
            )
        if len(self.symbols) != n:
            raise DataError(
                "len(Panel.symbols) 必须等于 N",
                context={"n_symbols": n, "len_symbols": len(self.symbols)},
            )
        if t >= 2 and not bool(np.all(np.diff(self.time) > 0)):
            raise DataError("Panel.time 必须严格递增（已排序且去重）", context={"T": t})

    # ── 属性 ────────────────────────────────────────────────────────────────

    @property
    def n_symbols(self) -> int:
        """截面/样本数 N。"""
        return int(self.open.shape[0])

    @property
    def n_bars(self) -> int:
        """时间长度 T。"""
        return int(self.open.shape[1])

    @property
    def fields(self) -> dict[str, np.ndarray]:
        """OHLCV 字段字典（规范顺序）。"""
        return {name: getattr(self, name) for name in PANEL_FIELDS}

    @property
    def fingerprint(self) -> str:
        """数据指纹 = sha256(OHLCV + time + symbols)（架构 §9.4）。"""
        return panel_fingerprint(self.fields, self.time, self.symbols)

    # ── 视图 / 转换 ─────────────────────────────────────────────────────────

    def slice_view(self, start: int, end: int) -> Panel:
        """按时间切片返回新 ``Panel``（``[start, end)``，端开；不改原对象）。"""
        t = self.n_bars
        lo = max(0, int(start))
        hi = min(t, int(end))
        if lo >= hi:
            raise DataError("slice_view 空切片", context={"start": start, "end": end, "T": t})
        sliced = {name: getattr(self, name)[:, lo:hi] for name in PANEL_FIELDS}
        return replace(
            self,
            time=self.time[lo:hi],
            meta=dict(self.meta),
            **sliced,
        )

    def to_raw_dict(self) -> dict[str, np.ndarray]:
        """导出为 ``{"open": [N,T], ..., "time": [T]}``（供特征层消费）。"""
        raw = self.fields
        raw["time"] = self.time
        return raw

    @classmethod
    def from_arrays(
        cls,
        fields: dict[str, np.ndarray],
        time: np.ndarray,
        *,
        symbols: tuple[str, ...],
        timeframe: str = "",
        market_profile_name: str = "",
        meta: dict[str, Any] | None = None,
    ) -> Panel:
        """从 ``{field: [N, T]}`` 构造 ``Panel``（缺失字段报错）。"""
        missing = [name for name in PANEL_FIELDS if name not in fields]
        if missing:
            raise DataError(f"Panel 缺少字段: {missing}", context={"missing": missing})
        return cls(
            open=fields["open"],
            high=fields["high"],
            low=fields["low"],
            close=fields["close"],
            volume=fields["volume"],
            time=time,
            symbols=tuple(symbols),
            timeframe=timeframe,
            market_profile_name=market_profile_name,
            meta=dict(meta) if meta else {},
        )

    def __repr__(self) -> str:  # pragma: no cover - 纯展示
        return (
            f"Panel(N={self.n_symbols}, T={self.n_bars}, symbols={self.symbols}, "
            f"timeframe={self.timeframe!r}, profile={self.market_profile_name!r})"
        )
