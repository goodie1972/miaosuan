# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""多重检验校正单测（M12）。"""

from __future__ import annotations

import pytest

from miaosuan.gate.multiple_testing import (
    benjamini_hochberg,
    bonferroni_threshold,
    deflated_sharpe_ratio,
)


def test_bonferroni_threshold() -> None:
    r = bonferroni_threshold(0.05, 100)
    assert r.threshold == pytest.approx(0.0005)
    assert r.n_trials == 100
    with pytest.raises(ValueError):
        bonferroni_threshold(1.5, 10)


def test_dsr_in_unit_interval() -> None:
    r = deflated_sharpe_ratio(0.02, n_trials=200, n_obs=1000)
    assert 0.0 <= r.dsr <= 1.0
    assert r.sr0 >= 0.0


def test_dsr_monotone_in_sharpe() -> None:
    lo = deflated_sharpe_ratio(0.01, n_trials=100, n_obs=1000)
    hi = deflated_sharpe_ratio(0.05, n_trials=100, n_obs=1000)
    assert hi.dsr > lo.dsr


def test_dsr_decreases_with_more_trials() -> None:
    few = deflated_sharpe_ratio(0.03, n_trials=10, n_obs=1000)
    many = deflated_sharpe_ratio(0.03, n_trials=5000, n_obs=1000)
    assert many.sr0 > few.sr0
    assert many.dsr < few.dsr


def test_dsr_passes_threshold() -> None:
    strong = deflated_sharpe_ratio(0.2, n_trials=50, n_obs=2000)
    assert strong.passes(0.95)
    weak = deflated_sharpe_ratio(0.0005, n_trials=5000, n_obs=500)
    assert not weak.passes(0.95)


def test_dsr_validation() -> None:
    with pytest.raises(ValueError):
        deflated_sharpe_ratio(0.01, n_trials=10, n_obs=1)
    with pytest.raises(ValueError):
        deflated_sharpe_ratio(0.01, n_trials=10, n_obs=100, sharpe_variance=-1.0)


def test_benjamini_hochberg() -> None:
    assert benjamini_hochberg([]) == ()
    rejected = benjamini_hochberg([0.001, 0.02, 0.5, 0.9], alpha=0.05)
    assert rejected[0] is True
    assert rejected[-1] is False
    with pytest.raises(ValueError):
        benjamini_hochberg([1.2], alpha=0.05)
