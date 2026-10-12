# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for full license text.

"""风险仓位（路线图 R3.1）：波动率目标仓位（纯 numpy，可注入可单测）。

口径（Sample E v1.1）：手数 = 波动预算 / 单手单 bar 美元波动 × Sharpe 权重，
向 ``lot_step`` 向下取整。**风险红线**：目标手数低于 ``min_lot`` 一律返回
0（不交易）——禁止向上取整到 min_lot，那会让弱信号被迫开出最小仓，
放大而非缩小风险。

单品种路线下「基准」即本品种自身的波动预算（组合级资金管理是非目标，
见路线图 §2.2）；全部规格取自 :class:`~miaosuan.market.profiles.BrokerSpec`
与画像成本模型，不硬编码。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:  # 仅类型提示
    from ..market.profiles import BrokerSpec

__all__ = ["RiskPolicy", "VolTargetPolicy", "atr", "vol_target_lots"]


@runtime_checkable
class RiskPolicy(Protocol):
    """风险仓位协议：信号 → 目标手数（带符号，多正空负）。"""

    def target_lots(
        self,
        *,
        signal: float,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        broker_spec: BrokerSpec,
    ) -> float: ...


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int) -> float:
    """平均真实波幅（Wilder 简化版：TR 的窗口均值，末值）。

    TR[t] = max(high-low, |high-prev_close|, |low-prev_close|)；
    首根无前收盘价，TR[0] = high-low。``window`` 取末 ``window`` 根 TR 均值。
    """
    h = np.asarray(high, dtype=np.float64)
    lo = np.asarray(low, dtype=np.float64)
    c = np.asarray(close, dtype=np.float64)
    if h.shape != lo.shape or h.shape != c.shape or h.size == 0:
        return 0.0
    prev_close = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - lo, np.maximum(np.abs(h - prev_close), np.abs(lo - prev_close)))
    w = max(1, min(int(window), tr.size))
    return float(np.mean(tr[-w:]))


def vol_target_lots(
    *,
    ref_lots: float,
    ref_dollar_vol: float,
    dollar_vol_per_lot: float,
    sharpe_weight: float,
    lot_step: float,
    min_lot: float,
) -> float:
    """波动率目标手数（Sample E v1.1，纯函数）。

    :param ref_lots: 基准手数（单品种路线 = 1.0，预算全部作用在本品种）。
    :param ref_dollar_vol: 基准单 bar 美元波动预算（RiskPolicy 配置）。
    :param dollar_vol_per_lot: 目标品种单手单 bar 美元波动（ATR×contract_size）。
    :param sharpe_weight: Sharpe 权重（调用方 clamp 到 [w_min, w_max]）。
    :param lot_step: 手数步长（来自 BrokerSpec）。
    :param min_lot: 最小手数（来自 BrokerSpec）。

    :returns: 向 ``lot_step`` 取整的手数；**低于 ``min_lot`` 返回 0**（不交易）。
    """
    if dollar_vol_per_lot <= 0 or lot_step <= 0:
        return 0.0
    raw = ref_lots * (ref_dollar_vol / dollar_vol_per_lot) * sharpe_weight
    if raw < min_lot:
        return 0.0  # 风险红线：弱信号不开最小仓
    return round(math.floor(raw / lot_step) * lot_step, 8)


@dataclass(frozen=True)
class VolTargetPolicy:
    """波动率目标仓位策略（:class:`RiskPolicy` 的第一实现）。

    :param risk_usd_per_bar: 单 bar 美元波动预算（"这一根 bar 我愿意承受多少
        美元波动"）；单品种路线下即本品种的预算。
    :param atr_window: ATR 回看窗口（bar 数）。
    :param w_min / w_max: Sharpe 权重的 clamp 区间。
    :param sharpe_weight: 当前权重（后续可由策略证据动态更新；静态配置起步）。
    """

    risk_usd_per_bar: float
    atr_window: int = 14
    w_min: float = 0.5
    w_max: float = 2.0
    sharpe_weight: float = 1.0

    def target_lots(
        self,
        *,
        signal: float,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        broker_spec: BrokerSpec,
    ) -> float:
        """信号 [-1,1] → 带符号目标手数（方向取信号符号，大小由波动目标定）。"""
        if signal == 0.0 or not np.isfinite(signal):
            return 0.0
        weight = float(min(max(self.sharpe_weight, self.w_min), self.w_max))
        atr_val = atr(high, low, close, self.atr_window)
        if atr_val <= 0.0:
            return 0.0
        dollar_vol_per_lot = atr_val * float(broker_spec.contract_size)
        lots = vol_target_lots(
            ref_lots=1.0,
            ref_dollar_vol=float(self.risk_usd_per_bar),
            dollar_vol_per_lot=dollar_vol_per_lot,
            sharpe_weight=weight,
            lot_step=float(broker_spec.lot_step),
            min_lot=float(broker_spec.min_lot),
        )
        if lots <= 0.0:
            return 0.0
        return lots if signal > 0 else -lots
