# -*- coding: utf-8 -*-
# Copyright (c) 2026 Original Author
# Copyright (c) 2026 Your Company
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""统一绩效指标 + 成本敏感性曲线（架构 §2 ``report/metrics.py``，M9）。

* **统一指标**：Sharpe / Sortino / Calmar / 最大回撤（MDD）/ 波动 / 换手 / 年化 / IC；
* **成本敏感性曲线**：在 ``base_cost_rate`` 的 **0.5x / 1x / 2x / 3x** 四档下分别给出
  Sharpe 与 MDD（架构 §6.4），供门禁（T04 ``gate/cost_curve.py``）与报告消费。

口径说明：

* 收益序列 ``pnl`` 为**逐 bar 净收益**（分数）；年化用 ``periods_per_year``（来自
  :class:`~miaosuan.market.profiles.FrozenMarketProfile`.bars_per_year）；
* ``max_drawdown`` 返回**正数**（峰值到谷底的相对跌幅）；
* 全为零/常数序列 → 指标返回 ``0.0``（不产 NaN），保证可排序。
* 支持**买卖不对称**成本（``sell_rate`` 缺省等于 ``base_cost_rate``）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "CostSensitivityResult",
    "DEFAULT_COST_MULTIPLIERS",
    "PerformanceMetrics",
    "annual_return",
    "calmar",
    "compute_metrics",
    "cost_sensitivity",
    "information_coefficient",
    "max_drawdown",
    "sharpe",
    "sortino",
    "turnover",
]

_EPS = 1e-12

#: 成本敏感性档位（架构 §6.4：0.5x / 1x / 2x / 3x）
DEFAULT_COST_MULTIPLIERS: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0)


def _as_1d(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(-1)


def _as_2d(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def annual_return(pnl: Any, periods_per_year: int) -> float:
    """年化收益 = 每 bar 平均收益 × ``periods_per_year``。"""
    arr = _as_1d(pnl)
    if arr.size == 0:
        return 0.0
    return float(arr.mean() * periods_per_year)


def volatility(pnl: Any, periods_per_year: int) -> float:
    """年化波动 = 每 bar 收益标准差 × √``periods_per_year``。"""
    arr = _as_1d(pnl)
    if arr.size < 2:
        return 0.0
    return float(arr.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe(pnl: Any, periods_per_year: int) -> float:
    """年化 Sharpe（无风险利率取 0）；标准差过小 → 0.0。"""
    arr = _as_1d(pnl)
    if arr.size < 2:
        return 0.0
    std = float(arr.std(ddof=1))
    if std < _EPS:
        return 0.0
    return float(arr.mean() / std * np.sqrt(periods_per_year))


def sortino(pnl: Any, periods_per_year: int) -> float:
    """年化 Sortino（下行标准差）；无下行样本 → 0.0。"""
    arr = _as_1d(pnl)
    if arr.size < 2:
        return 0.0
    downside = arr[arr < 0.0]
    if downside.size < 2:
        return 0.0
    dstd = float(downside.std(ddof=1))
    if dstd < _EPS:
        return 0.0
    return float(arr.mean() / dstd * np.sqrt(periods_per_year))


def max_drawdown(pnl: Any) -> float:
    """最大回撤（正数，基于 ``1 + cumsum(pnl)`` 权益曲线）。"""
    arr = _as_1d(pnl)
    if arr.size == 0:
        return 0.0
    equity = 1.0 + np.cumsum(arr)
    peak = np.maximum.accumulate(equity)
    denom = np.maximum(np.abs(peak), _EPS)
    drawdown = (peak - equity) / denom
    return float(np.max(drawdown))


def calmar(pnl: Any, periods_per_year: int) -> float:
    """年化收益 / 最大回撤；无回撤 → 0.0。"""
    mdd = max_drawdown(pnl)
    if mdd < _EPS:
        return 0.0
    return float(annual_return(pnl, periods_per_year) / mdd)


def turnover(position: Any) -> float:
    """平均换手：每品种逐 bar ``|Δ仓位|`` 之和的截面均值。"""
    pos = _as_2d(position)
    if pos.shape[-1] == 0:
        return 0.0
    prev = np.zeros_like(pos)
    if pos.shape[-1] > 1:
        prev[:, 1:] = pos[:, :-1]
    delta = np.abs(pos - prev)
    return float(delta.sum(axis=1).mean())


def information_coefficient(factor: Any, target: Any) -> float:
    """IC：逐品种时间维 Pearson 相关的截面均值（样本不足/常数 → 0.0）。"""
    f = _as_2d(factor)
    t = _as_2d(target)
    n = min(f.shape[0], t.shape[0])
    t_len = min(f.shape[1], t.shape[1])
    if n == 0 or t_len < 2:
        return 0.0
    ics: list[float] = []
    for i in range(n):
        fv = f[i, :t_len]
        tv = t[i, :t_len]
        mask = np.isfinite(fv) & np.isfinite(tv)
        if int(mask.sum()) < 2:
            continue
        a, b = fv[mask], tv[mask]
        if float(a.std()) < _EPS or float(b.std()) < _EPS:
            continue
        corr = np.corrcoef(a, b)[0, 1]
        if np.isfinite(corr):
            ics.append(float(corr))
    return float(np.mean(ics)) if ics else 0.0


@dataclass(frozen=True)
class PerformanceMetrics:
    """统一绩效指标（单一次运行）。"""

    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    annual_return: float
    volatility: float
    turnover: float
    n_bars: int

    def to_dict(self) -> dict[str, float]:
        """序列化为 JSON 友好字典。"""
        return {
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "calmar": self.calmar,
            "max_drawdown": self.max_drawdown,
            "annual_return": self.annual_return,
            "volatility": self.volatility,
            "turnover": self.turnover,
            "n_bars": float(self.n_bars),
        }


def compute_metrics(
    pnl: Any,
    *,
    periods_per_year: int,
    position: Any | None = None,
) -> PerformanceMetrics:
    """从逐 bar 净收益计算统一指标（可选传入仓位以计算换手）。"""
    arr = _as_1d(pnl)
    return PerformanceMetrics(
        sharpe=sharpe(arr, periods_per_year),
        sortino=sortino(arr, periods_per_year),
        calmar=calmar(arr, periods_per_year),
        max_drawdown=max_drawdown(arr),
        annual_return=annual_return(arr, periods_per_year),
        volatility=volatility(arr, periods_per_year),
        turnover=turnover(position) if position is not None else 0.0,
        n_bars=int(arr.size),
    )


@dataclass(frozen=True)
class CostSensitivityResult:
    """成本敏感性曲线结果（0.5x / 1x / 2x / 3x 四档）。

    :param multipliers: 倍率档位（升序）。
    :param by_level: ``{倍率: PerformanceMetrics}``。
    :param baseline_cost_rate: 1x 档的单边成本率。
    """

    multipliers: tuple[float, ...]
    by_level: dict[float, PerformanceMetrics]
    baseline_cost_rate: float

    def sharpes(self) -> dict[float, float]:
        """各档 Sharpe。"""
        return {m: self.by_level[m].sharpe for m in self.multipliers}

    def max_drawdowns(self) -> dict[float, float]:
        """各档最大回撤（正数）。"""
        return {m: self.by_level[m].max_drawdown for m in self.multipliers}

    def to_report(self) -> dict[str, Any]:
        """产出报告友好的嵌套字典（含四档 Sharpe 与 MDD）。"""
        return {
            "baseline_cost_rate": self.baseline_cost_rate,
            "levels": list(self.multipliers),
            "by_level": [
                {
                    "multiplier": m,
                    "cost_rate": round(self.baseline_cost_rate * m, 12),
                    "sharpe": self.by_level[m].sharpe,
                    "max_drawdown": self.by_level[m].max_drawdown,
                    "sortino": self.by_level[m].sortino,
                    "calmar": self.by_level[m].calmar,
                }
                for m in self.multipliers
            ],
        }


def cost_sensitivity(
    position: Any,
    ret: Any,
    *,
    base_cost_rate: float,
    periods_per_year: int,
    multipliers: tuple[float, ...] = DEFAULT_COST_MULTIPLIERS,
    sell_rate: float | None = None,
) -> CostSensitivityResult:
    """成本敏感性曲线：在 ``base_cost_rate × multiplier`` 下重算 Sharpe / MDD。

    成本按**换手**计：``net = position * ret - (buy_rate * Δpos⁺ + sell_rate * Δpos⁻)``；
    支持买卖不对称（``sell_rate`` 缺省等于 ``base_cost_rate``）。

    :param position: ``[N, T]`` 目标仓位。
    :param ret: ``[N, T]`` 标的收益。
    :param base_cost_rate: 1x 档的单边买入成本率。
    :param periods_per_year: 年化因子（来自 MarketProfile.bars_per_year）。
    :param multipliers: 档位倍率（默认 0.5/1/2/3）。
    :param sell_rate: 1x 档的单边卖出成本率（缺省 = ``base_cost_rate``）。
    """
    pos = _as_2d(position)
    r = _as_2d(ret)
    if pos.shape != r.shape:
        raise ValueError(f"position 与 ret 形状必须一致：{pos.shape} vs {r.shape}")
    buy0 = float(base_cost_rate)
    sell0 = buy0 if sell_rate is None else float(sell_rate)

    prev = np.zeros_like(pos)
    if pos.shape[-1] > 1:
        prev[:, 1:] = pos[:, :-1]
    delta = pos - prev
    increase = np.clip(delta, 0.0, None)
    decrease = np.clip(-delta, 0.0, None)
    gross = pos * r

    by_level: dict[float, PerformanceMetrics] = {}
    for m in multipliers:
        level = float(m)
        net = gross - (buy0 * level) * increase - (sell0 * level) * decrease
        by_level[level] = compute_metrics(
            net, periods_per_year=periods_per_year, position=pos
        )

    return CostSensitivityResult(
        multipliers=tuple(float(m) for m in multipliers),
        by_level=by_level,
        baseline_cost_rate=buy0,
    )
