"""成本敏感性曲线与 2x 成本闸单测（M12）。"""

from __future__ import annotations

import numpy as np

from miaosuan.gate.cost_curve import cost_gate, evaluate_cost_curve


def _edge_arrays(seed: int = 0, n: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """构造「仓位 = 符号(当期收益)」的强正边际样本（毛收益 = |ret|）。"""
    rng = np.random.default_rng(seed)
    ret = rng.normal(0.0, 0.002, (1, n))
    position = np.sign(ret)
    return position.astype(np.float64), ret.astype(np.float64)


def test_cost_curve_levels() -> None:
    position, ret = _edge_arrays()
    curve = evaluate_cost_curve(position, ret, base_cost_rate=0.0002, periods_per_year=6240)
    assert set(curve.multipliers) == {0.5, 1.0, 2.0, 3.0}
    assert set(curve.sharpes()) == {0.5, 1.0, 2.0, 3.0}
    assert curve.base_cost_rate == 0.0002
    assert {curve.max_drawdown_at(m) >= 0.0 for m in curve.multipliers} == {True}


def test_cost_curve_sharpe_monotone_decreasing_in_cost() -> None:
    position, ret = _edge_arrays()
    curve = evaluate_cost_curve(position, ret, base_cost_rate=0.0005, periods_per_year=6240)
    s05 = curve.sharpe_at(0.5)
    s1 = curve.sharpe_at(1.0)
    s2 = curve.sharpe_at(2.0)
    s3 = curve.sharpe_at(3.0)
    assert s05 >= s1 >= s2 >= s3


def test_cost_curve_unknown_multiplier_zero() -> None:
    position, ret = _edge_arrays()
    curve = evaluate_cost_curve(position, ret, base_cost_rate=0.0002, periods_per_year=6240)
    assert curve.sharpe_at(9.0) == 0.0
    assert curve.max_drawdown_at(9.0) == 0.0


def test_cost_gate_survives_low_cost() -> None:
    position, ret = _edge_arrays()
    passed, sharpe_2x = cost_gate(
        position, ret, base_cost_rate=0.0001, periods_per_year=6240, required_multiplier=2.0
    )
    assert isinstance(passed, bool)
    assert passed is True
    assert sharpe_2x > 0.0


def test_cost_gate_fails_at_high_cost() -> None:
    position, ret = _edge_arrays()
    # 极高成本 → 2x 档必然为负 → 拦截
    passed, sharpe_2x = cost_gate(
        position, ret, base_cost_rate=0.05, periods_per_year=6240, required_multiplier=2.0
    )
    assert passed is False
    assert sharpe_2x <= 0.0


def test_cost_curve_survives_helper() -> None:
    position, ret = _edge_arrays()
    curve = evaluate_cost_curve(position, ret, base_cost_rate=0.0001, periods_per_year=6240)
    assert curve.survives(multiplier=2.0, min_sharpe=0.0) is True
    assert curve.survives(multiplier=3.0, min_sharpe=1e9) is False


def test_cost_curve_to_dict() -> None:
    position, ret = _edge_arrays()
    curve = evaluate_cost_curve(position, ret, base_cost_rate=0.0002, periods_per_year=6240)
    d = curve.to_dict()
    assert d["base_cost_rate"] == 0.0002
    assert set(d["sharpe_by_level"]) == {"0.5", "1.0", "2.0", "3.0"}  # type: ignore[arg-type]
