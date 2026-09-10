"""成本敏感性曲线与 2x 成本闸（架构 §6.4，M12）。

**理念**：一个因子若在「1x 成本」下漂亮、在「2x 成本」下就死，则它是「捡硬币于推土机前」，
不可部署。本模块在 ``base_cost_rate`` 的 **0.5x / 1x / 2x / 3x** 四档下重算 Sharpe / MDD
（复用 :func:`miaosuan.report.metrics.cost_sensitivity`），并给出「2x 成本 Sharpe 闸」。

年化因子 ``periods_per_year`` 由调用方从 :class:`~miaosuan.market.profiles.FrozenMarketProfile`
的 ``bars_per_year`` 注入 —— 本模块**不硬编码 6240**。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..report.metrics import DEFAULT_COST_MULTIPLIERS, CostSensitivityResult, cost_sensitivity

__all__ = [
    "CostCurveResult",
    "cost_gate",
    "evaluate_cost_curve",
]


@dataclass(frozen=True)
class CostCurveResult:
    """成本敏感性曲线结果（包裹 :class:`~miaosuan.report.metrics.CostSensitivityResult`）。

    :param inner: 统一指标层的成本敏感性结果。
    :param base_cost_rate: 1x 档单边成本率。
    """

    inner: CostSensitivityResult
    base_cost_rate: float

    @property
    def multipliers(self) -> tuple[float, ...]:
        """档位倍率（升序）。"""
        return self.inner.multipliers

    def sharpe_at(self, multiplier: float) -> float:
        """指定倍率档的 Sharpe；未知倍率返回 ``0.0``。"""
        metrics = self.inner.by_level.get(float(multiplier))
        return 0.0 if metrics is None else float(metrics.sharpe)

    def max_drawdown_at(self, multiplier: float) -> float:
        """指定倍率档的最大回撤（正数）；未知倍率返回 ``0.0``。"""
        metrics = self.inner.by_level.get(float(multiplier))
        return 0.0 if metrics is None else float(metrics.max_drawdown)

    def sharpes(self) -> dict[float, float]:
        """各档 Sharpe。"""
        return self.inner.sharpes()

    def survives(self, *, multiplier: float = 2.0, min_sharpe: float = 0.0) -> bool:
        """在指定倍率档下 Sharpe 是否仍 ``> min_sharpe``（默认 2x 档 > 0）。"""
        return bool(self.sharpe_at(multiplier) > float(min_sharpe))

    def to_dict(self) -> dict[str, object]:
        """JSON 友好字典（含各档 Sharpe / MDD）。"""
        return {
            "base_cost_rate": self.base_cost_rate,
            "levels": list(self.multipliers),
            "sharpe_by_level": {str(m): self.sharpe_at(m) for m in self.multipliers},
            "max_drawdown_by_level": {str(m): self.max_drawdown_at(m) for m in self.multipliers},
        }


def evaluate_cost_curve(
    position: np.ndarray,
    ret: np.ndarray,
    *,
    base_cost_rate: float,
    periods_per_year: int,
    multipliers: tuple[float, ...] = DEFAULT_COST_MULTIPLIERS,
    sell_rate: float | None = None,
) -> CostCurveResult:
    """计算成本敏感性曲线（0.5x/1x/2x/3x 的 Sharpe 与 MDD）。

    :param position: ``[N, T]`` 目标仓位。
    :param ret: ``[N, T]`` 标的收益。
    :param base_cost_rate: 1x 档单边买入成本率。
    :param periods_per_year: 年化因子（来自 profile ``bars_per_year``）。
    :param multipliers: 档位倍率（默认 0.5/1/2/3）。
    :param sell_rate: 1x 档单边卖出成本率（缺省 = ``base_cost_rate``）。
    """
    inner = cost_sensitivity(
        position,
        ret,
        base_cost_rate=float(base_cost_rate),
        periods_per_year=int(periods_per_year),
        multipliers=multipliers,
        sell_rate=sell_rate,
    )
    return CostCurveResult(inner=inner, base_cost_rate=float(base_cost_rate))


def cost_gate(
    position: np.ndarray,
    ret: np.ndarray,
    *,
    base_cost_rate: float,
    periods_per_year: int,
    required_multiplier: float = 2.0,
    min_sharpe: float = 0.0,
    multipliers: tuple[float, ...] = DEFAULT_COST_MULTIPLIERS,
    sell_rate: float | None = None,
) -> tuple[bool, float]:
    """2x 成本闸：返回 ``(是否通过, 该档 Sharpe)``。

    ``通过`` 当且仅当 ``required_multiplier`` 档 Sharpe ``> min_sharpe``（默认 2x > 0）。
    """
    curve = evaluate_cost_curve(
        position,
        ret,
        base_cost_rate=base_cost_rate,
        periods_per_year=periods_per_year,
        multipliers=multipliers,
        sell_rate=sell_rate,
    )
    sharpe = curve.sharpe_at(required_multiplier)
    return bool(sharpe > float(min_sharpe)), float(sharpe)
