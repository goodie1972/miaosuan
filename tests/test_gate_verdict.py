# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""门禁综合 verdict 单测（M12 / 验收 #4）。

验收 #4：**固定 fixture → 稳定 verdict 快照**。此处以固定数值输入断言 ``snapshot()`` 确定性。
"""

from __future__ import annotations

from miaosuan.gate.verdict import (
    STATUS_BLOCKED,
    STATUS_DEPLOYABLE,
    STATUS_RESEARCH_ONLY,
    GateThresholds,
    evaluate_verdict,
)


def test_verdict_deployable() -> None:
    v = evaluate_verdict(
        val_score=2.7,
        train_score=3.0,
        dsr=0.99,
        sharpe_1x=1.5,
        sharpe_2x=1.2,
        wf_win_rate=0.8,
        n_trials=500,
        n_obs=2000,
    )
    assert v.status == STATUS_DEPLOYABLE
    assert v.deployable is True
    assert v.hard_failures == ()
    assert v.soft_failures == ()


def test_verdict_research_only_on_decay() -> None:
    v = evaluate_verdict(
        val_score=2.7,
        train_score=6.0,  # decay 0.45 < 0.5 → 过拟合软失败
        dsr=0.99,
        sharpe_1x=1.5,
        sharpe_2x=1.2,
        wf_win_rate=0.8,
    )
    assert v.status == STATUS_RESEARCH_ONLY
    assert "decay_ratio<0.5" in v.soft_failures


def test_verdict_research_only_on_wf_win_rate() -> None:
    v = evaluate_verdict(
        val_score=2.7,
        train_score=3.0,
        dsr=0.99,
        sharpe_1x=1.5,
        sharpe_2x=1.2,
        wf_win_rate=0.4,  # < 0.6
    )
    assert v.status == STATUS_RESEARCH_ONLY
    assert "wf_win_rate<0.6" in v.soft_failures


def test_verdict_research_only_on_soft_dsr() -> None:
    v = evaluate_verdict(
        val_score=2.7,
        train_score=3.0,
        dsr=0.7,  # >= 0.5 但 < 0.95
        sharpe_1x=1.5,
        sharpe_2x=1.2,
        wf_win_rate=0.8,
    )
    assert v.status == STATUS_RESEARCH_ONLY
    assert "dsr<0.95" in v.soft_failures


def test_verdict_blocked_on_nonpositive_val() -> None:
    v = evaluate_verdict(
        val_score=-0.1, train_score=1.0, dsr=0.99, sharpe_1x=1.0, sharpe_2x=0.5, wf_win_rate=0.9
    )
    assert v.status == STATUS_BLOCKED
    assert "val_score<=0" in v.hard_failures


def test_verdict_blocked_on_cost_death() -> None:
    v = evaluate_verdict(
        val_score=2.7, train_score=3.0, dsr=0.99, sharpe_1x=1.0, sharpe_2x=-0.2, wf_win_rate=0.9
    )
    assert v.status == STATUS_BLOCKED
    assert "sharpe_2x<=0" in v.hard_failures
    assert v.soft_failures == (), "硬失败时不再评估软失败"


def test_verdict_blocked_on_dsr_noise() -> None:
    v = evaluate_verdict(
        val_score=2.7, train_score=3.0, dsr=0.3, sharpe_1x=1.0, sharpe_2x=0.5, wf_win_rate=0.9
    )
    assert v.status == STATUS_BLOCKED
    assert "dsr<0.5" in v.hard_failures


def test_verdict_snapshot_deterministic() -> None:
    kwargs = {
        "val_score": 2.701234567,
        "train_score": 2.9,
        "dsr": 0.987654321,
        "sharpe_1x": 1.111111,
        "sharpe_2x": 1.222222,
        "wf_win_rate": 0.75,
        "n_trials": 512,
        "n_obs": 4096,
    }
    a = evaluate_verdict(**kwargs).snapshot()
    b = evaluate_verdict(**kwargs).snapshot()
    assert a == b
    # 键排序（JSON sort_keys）→ 稳定前缀
    assert a.startswith('{"decay_ratio"')


def test_verdict_threshold_override() -> None:
    th = GateThresholds(min_decay_ratio=0.3)
    v = evaluate_verdict(
        val_score=2.7, train_score=6.0, dsr=0.99, sharpe_1x=1.5, sharpe_2x=1.2, wf_win_rate=0.8,
        thresholds=th,
    )
    assert v.status == STATUS_DEPLOYABLE  # 放宽衰减阈值后通过


def test_verdict_to_dict_rounds() -> None:
    v = evaluate_verdict(
        val_score=2.70123456789, train_score=2.9, dsr=0.9, sharpe_1x=1.0, sharpe_2x=0.5, wf_win_rate=0.75
    )
    d = v.to_dict()
    assert d["val_score"] == 2.701235
