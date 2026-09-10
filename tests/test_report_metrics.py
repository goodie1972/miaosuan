"""统一指标 + 成本敏感性曲线单测（M9）。

（批量）验收：报告含 **0.5x / 1x / 2x / 3x 四档 Sharpe 与 MDD**。
"""

from __future__ import annotations

import numpy as np
import pytest

from miaosuan.report.metrics import (
    DEFAULT_COST_MULTIPLIERS,
    annual_return,
    calmar,
    compute_metrics,
    cost_sensitivity,
    information_coefficient,
    max_drawdown,
    sharpe,
    sortino,
    turnover,
    volatility,
)

PPY = 6240


# ── 基础指标 ────────────────────────────────────────────────────────────────


def test_sharpe_and_sortino_basic() -> None:
    rng = np.random.default_rng(5)
    pnl = rng.normal(1e-4, 1e-3, 2000)
    assert np.isfinite(sharpe(pnl, PPY))
    assert np.isfinite(sortino(pnl, PPY))
    assert volatility(pnl, PPY) > 0
    assert abs(annual_return(pnl, PPY) - pnl.mean() * PPY) < 1e-12


def test_sharpe_zero_for_constant_series() -> None:
    assert sharpe(np.full(100, 0.001), PPY) == 0.0
    assert sortino(np.full(100, 0.001), PPY) == 0.0
    assert sharpe(np.array([0.001]), PPY) == 0.0


def test_max_drawdown_known_value() -> None:
    pnl = np.array([0.1, -0.5, 0.2])
    # equity = [1.1, 0.6, 0.8]；peak = [1.1, 1.1, 1.1]；max dd = 0.5/1.1
    assert max_drawdown(pnl) == pytest.approx(0.5 / 1.1, rel=1e-9)
    assert max_drawdown(np.array([0.1, 0.2, 0.3])) == 0.0
    assert max_drawdown(np.array([])) == 0.0


def test_calmar_zero_when_no_drawdown() -> None:
    assert calmar(np.array([0.1, 0.2, 0.3]), PPY) == 0.0
    assert np.isfinite(calmar(np.array([0.1, -0.05, 0.02]), PPY))


def test_turnover() -> None:
    position = np.array([[0.0, 1.0, 1.0, 0.0]], dtype=np.float64)
    assert turnover(position) == pytest.approx(2.0)


def test_information_coefficient_perfect() -> None:
    target = np.array([[0.1, -0.2, 0.3, 0.4]], dtype=np.float64)
    assert information_coefficient(target, target) == pytest.approx(1.0)
    assert information_coefficient(target, -target) == pytest.approx(-1.0)
    assert information_coefficient(np.zeros((1, 4)), target) == 0.0


def test_compute_metrics_dict() -> None:
    m = compute_metrics(np.array([0.001, -0.0005, 0.002]), periods_per_year=PPY)
    d = m.to_dict()
    assert set(d) == {
        "sharpe", "sortino", "calmar", "max_drawdown",
        "annual_return", "volatility", "turnover", "n_bars",
    }
    assert d["n_bars"] == 3.0


# ── 成本敏感性曲线 ──────────────────────────────────────────────────────────


def _high_turnover_case() -> tuple[np.ndarray, np.ndarray]:
    """高换手 + 正 edge 的仓位/收益（成本越高 Sharpe 越低）。"""
    t = 2000
    rng = np.random.default_rng(123)
    position = np.empty((1, t), dtype=np.float64)
    position[0, ::2] = 1.0
    position[0, 1::2] = -1.0
    ret = position * 5e-4 + rng.normal(0.0, 1e-4, (1, t))
    return position, ret


def test_cost_sensitivity_four_levels() -> None:
    position, ret = _high_turnover_case()
    result = cost_sensitivity(
        position, ret, base_cost_rate=0.0003, periods_per_year=PPY
    )
    assert result.multipliers == DEFAULT_COST_MULTIPLIERS == (0.5, 1.0, 2.0, 3.0)
    assert set(result.by_level) == {0.5, 1.0, 2.0, 3.0}


def test_cost_sensitivity_report_contains_sharpe_and_mdd() -> None:
    position, ret = _high_turnover_case()
    report = cost_sensitivity(
        position, ret, base_cost_rate=0.0003, periods_per_year=PPY
    ).to_report()
    assert report["levels"] == [0.5, 1.0, 2.0, 3.0]
    assert len(report["by_level"]) == 4
    for entry in report["by_level"]:
        assert "sharpe" in entry and "max_drawdown" in entry
        assert np.isfinite(entry["sharpe"])
        assert entry["max_drawdown"] >= 0.0
    # 成本越高 → Sharpe 越低
    sharpes = [e["sharpe"] for e in report["by_level"]]
    assert sharpes[0] >= sharpes[-1], f"成本敏感性方向异常: {sharpes}"


def test_cost_sensitivity_zero_cost_matches_gross() -> None:
    position, ret = _high_turnover_case()
    result = cost_sensitivity(position, ret, base_cost_rate=0.0, periods_per_year=PPY)
    gross_sharpe = sharpe(position * ret, PPY)
    assert all(m.sharpe == pytest.approx(gross_sharpe) for m in result.by_level.values())


def test_cost_sensitivity_asymmetric() -> None:
    position, ret = _high_turnover_case()
    sym = cost_sensitivity(position, ret, base_cost_rate=0.0003, periods_per_year=PPY)
    asym = cost_sensitivity(
        position, ret, base_cost_rate=0.0003, sell_rate=0.0013, periods_per_year=PPY
    )
    # 卖出成本更高 → Sharpe 不高于对称情形
    assert asym.sharpes()[1.0] <= sym.sharpes()[1.0] + 1e-9


def test_cost_sensitivity_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        cost_sensitivity(
            np.zeros((1, 5)), np.zeros((1, 4)), base_cost_rate=0.0, periods_per_year=PPY
        )
