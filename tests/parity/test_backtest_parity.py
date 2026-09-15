# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""M5 backtest 数值对拍：妙算 numpy vs 冻结 AM ``model_core/backtest.py``。

AM 运行态：``MT5Backtest(cost_rate=0.0003, periods_per_year=6240)``，``REWARD_MODE="ftmo"``
（根目录 config / ModelConfig 当前取值）。妙算以参数注入同值。
"""

from __future__ import annotations

import numpy as np
import pytest

from miaosuan.core.backtest import MT5Backtest, estimate_periods_per_year

pytestmark = pytest.mark.parity

_ATOL = 1e-4
_RTOL = 1e-4


def _bt() -> MT5Backtest:
    return MT5Backtest(cost_rate=0.0003, periods_per_year=6240, reward_mode="ftmo")


@pytest.mark.parametrize("name", ("single", "multi5", "short"))
def test_evaluate_parity(name: str, m5_npz: dict, m5_meta: dict) -> None:
    factors = m5_npz[f"bt_factors__{name}"]
    target = m5_npz[f"bt_target__{name}"]
    score, mean_oos = _bt().evaluate(factors, {}, target)
    exp = m5_meta["backtest"][name]
    assert abs(score - exp["score"]) <= _ATOL + _RTOL * abs(exp["score"])
    assert abs(mean_oos - exp["mean_oos"]) <= _ATOL + _RTOL * abs(exp["mean_oos"])


@pytest.mark.parametrize("name", ("single", "multi5", "short"))
def test_evaluate_fold_parity(name: str, m5_npz: dict, m5_meta: dict) -> None:
    factors = m5_npz[f"bt_factors__{name}"]
    target = m5_npz[f"bt_target__{name}"]
    exp = m5_meta["backtest"][name]
    split = int(exp["fold_split"])
    tr, vl = _bt().evaluate_fold(factors, target, 0, split, split, factors.shape[1])
    assert abs(tr - exp["fold_train"]) <= _ATOL + _RTOL * abs(exp["fold_train"])
    assert abs(vl - exp["fold_val"]) <= _ATOL + _RTOL * abs(exp["fold_val"])


def test_reward_mode_branches() -> None:
    """三种 reward_mode 分支均可运行且产出有限值（接口预留，默认 ftmo）。"""
    rng = np.random.default_rng(99)
    factors = rng.normal(0, 1, (3, 400)).astype(np.float32)
    target = rng.normal(0, 1e-3, (3, 400)).astype(np.float32)
    for mode in ("ftmo", "forex", "default"):
        bt = MT5Backtest(cost_rate=0.0003, periods_per_year=6240, reward_mode=mode)
        score, _ = bt.evaluate(factors, {}, target)
        assert np.isfinite(score)


_SECONDS_PER_YEAR = 365.25 * 86400.0


def test_estimate_periods_per_year_span_based() -> None:
    """年化因子 = 样本数 / 时间跨度（年），与 AM ``estimate_periods_per_year`` 同公式。

    * N 根 bar 恰跨 1 年 → ≈ N 根/年（数据驱动，自动适配市场/周期）；
    * 连续小时序列（24h × 365.25）→ ≈ 8766 根/年（H1 连续市场，含周末）；
      注意：AM 的 ``6240`` 仅是「时间戳不可用」时的**兜底**（24 × 260 交易日），
      并非 H1 的固定值。
    """
    n = 6240
    t = np.round(np.linspace(0, _SECONDS_PER_YEAR, n)).astype(np.int64)
    assert estimate_periods_per_year(t) == 6240

    t_hourly = np.arange(8766, dtype=np.int64) * 3600
    ppy_hourly = estimate_periods_per_year(t_hourly)
    assert 8700 <= ppy_hourly <= 8800, f"连续小时序列年化={ppy_hourly}（期望 ≈8766）"


def test_estimate_periods_per_year_invalid_falls_back() -> None:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert estimate_periods_per_year(np.array([0], dtype=np.int64)) == 6240
