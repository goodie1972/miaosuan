# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""M5 signal 数值对拍：妙算 numpy vs 冻结 AM 的 ``strategy_manager/signal.py``。

连续仓位 = ``tanh(factor)`` 后套中性带（``|pos| < min_trade_exposure`` → 0）；并校验妙算新增
的 ``long_only`` 预留开关（默认 False 与 AM 一致）。
"""

from __future__ import annotations

import numpy as np
import pytest

from miaosuan.core.signal import (
    MIN_TRADE_EXPOSURE,
    SignalMapper,
    compute_target_positions,
    compute_target_positions_stateless,
    reconcile_action,
    target_to_direction,
)

pytestmark = pytest.mark.parity

_ATOL = 1e-6


@pytest.mark.parametrize(
    "name",
    ("n1_t600", "n4_t320", "edge"),
)
def test_signal_parity(name: str, m5_npz: dict) -> None:
    factors = m5_npz[f"sig_in__{name}"]
    got = compute_target_positions(factors)
    exp = m5_npz[f"sig_out__{name}"]
    assert got.shape == exp.shape
    assert float(np.max(np.abs(got - exp))) <= _ATOL


def test_signal_stateless_equals_stateful(m5_npz: dict) -> None:
    factors = m5_npz["sig_in__n1_t600"]
    a = compute_target_positions(factors)
    b = compute_target_positions_stateless(factors)
    assert np.array_equal(a, b)


def test_neutral_band_semantics() -> None:
    """|tanh(factor)| < 0.05 → 0；≥ 0.05 → 保留。"""
    pos = np.array([0.0, 0.02, 0.0499, 0.0501, 0.3, -0.02, -0.0501, -0.3], dtype=np.float64)
    factors = np.arctanh(pos).astype(np.float32)
    got = compute_target_positions(factors)
    expected = np.where(np.abs(np.tanh(factors)) >= MIN_TRADE_EXPOSURE, np.tanh(factors), 0.0)
    assert np.allclose(got, expected, atol=1e-6)
    assert got[0] == 0.0 and got[1] == 0.0 and got[2] == 0.0  # 带内 → 0
    assert got[3] != 0.0 and got[4] != 0.0  # 带外 → 保留


def test_long_only_reserved_switch() -> None:
    """long_only=True 时仓位无负值（架构 §T02 验收 4 的预留开关）。"""
    factors = np.linspace(-3, 3, 200).astype(np.float32)
    pos = compute_target_positions(factors, long_only=True)
    assert float(pos.min()) >= 0.0
    # 与 AM 默认（long_only=False）在有负值处不同
    pos_default = compute_target_positions(factors, long_only=False)
    assert float(pos_default.min()) < 0.0


def test_signal_mapper_matches_function() -> None:
    factors = np.random.default_rng(7).normal(0, 1, (1, 400)).astype(np.float32)
    mapper = SignalMapper()
    assert np.array_equal(mapper.to_position(factors), compute_target_positions(factors))
    with pytest.raises(Exception):  # noqa: B017 - 非法 position_fn 抛 ConfigError
        SignalMapper(position_fn="sigmoid").to_position(factors)


def test_target_to_direction_and_reconcile() -> None:
    assert target_to_direction(0.5) == 1
    assert target_to_direction(-0.5) == -1
    assert target_to_direction(0.01) == 0
    assert reconcile_action(1, 1) == "HOLD"
    assert reconcile_action(0, 1) == "OPEN_LONG"
    assert reconcile_action(0, -1) == "OPEN_SHORT"
    assert reconcile_action(1, 0) == "CLOSE"
    assert reconcile_action(1, -1) == "REVERSE_TO_SHORT"
    assert reconcile_action(-1, 1) == "REVERSE_TO_LONG"
