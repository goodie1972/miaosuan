# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""三档预算 / 墙钟硬约束 / 分阶段早停单测（M11）。"""

from __future__ import annotations

import numpy as np
import pytest

from miaosuan.config import GASearchConfig
from miaosuan.errors import ConfigError
from miaosuan.search.budget import (
    STOP_CONVERGED,
    STOP_EARLY_STOP,
    STOP_MAX_GENERATIONS,
    STOP_RUNNING,
    STOP_WALL_CLOCK,
    Budget,
)
from miaosuan.search.ga import EvalResult, RpnGA
from miaosuan.search.rpn import random_feasible

# ── 档位参数表（§6.2）───────────────────────────────────────────────────────


def test_profile_table_values() -> None:
    quick = Budget.from_profile("quick")
    standard = Budget.from_profile("standard")
    deep = Budget.from_profile("deep")

    assert (quick.wall_clock_hours, standard.wall_clock_hours, deep.wall_clock_hours) == (0.5, 2.0, 8.0)
    assert (quick.max_generations, standard.max_generations, deep.max_generations) == (60, 300, 1200)
    assert (quick.pop_size, standard.pop_size, deep.pop_size) == (96, 256, 384)
    assert (quick.elite_size, standard.elite_size, deep.elite_size) == (12, 32, 48)
    assert (quick.min_hamming, standard.min_hamming, deep.min_hamming) == (2, 2, 2)
    assert (quick.island_count, standard.island_count, deep.island_count) == (1, 4, 4)
    assert quick.islands_enabled is False
    assert standard.islands_enabled is True


def test_default_profile_is_standard() -> None:
    assert Budget().profile == "standard"
    assert Budget.from_profile().profile == "standard"


def test_unknown_profile_raises() -> None:
    with pytest.raises(ConfigError):
        Budget.from_profile("ultra")


def test_wall_clock_seconds() -> None:
    assert Budget.from_profile("standard").wall_clock_seconds == 7200.0


def test_budget_validation() -> None:
    with pytest.raises(ConfigError):
        Budget(wall_clock_hours=0.0)
    with pytest.raises(ConfigError):
        Budget(pop_size=8, elite_size=16)


# ── 早停状态机 ──────────────────────────────────────────────────────────────


def test_stop_wall_clock_has_priority() -> None:
    b = Budget.from_profile("quick")
    b.reset()
    assert b.should_stop(0, 0.0, elapsed_seconds=10_000.0) is True
    assert b.stop_reason() == STOP_WALL_CLOCK


def test_stop_max_generations() -> None:
    b = Budget(profile="quick", wall_clock_hours=1000.0, max_generations=5)
    b.reset()
    for g in range(4):
        assert b.should_stop(g, float(g), 0.0) is False
    assert b.should_stop(4, 4.0, 0.0) is True
    assert b.stop_reason() == STOP_MAX_GENERATIONS


def test_stop_early_stop_after_explore() -> None:
    b = Budget(
        profile="quick",
        wall_clock_hours=1000.0,
        max_generations=1000,
        patience=3,
        min_delta=0.005,
        explore_generations=0,
    )
    b.reset()
    reasons = []
    for g in range(10):
        stopped = b.should_stop(g, 1.0, 0.0)  # 恒定不变 → 无改进
        reasons.append(b.stop_reason())
        if stopped:
            break
    assert reasons[-1] == STOP_EARLY_STOP
    assert STOP_RUNNING in reasons


def test_stop_converged_when_collapsed() -> None:
    b = Budget(
        profile="quick",
        wall_clock_hours=1000.0,
        max_generations=1000,
        patience=3,
        min_delta=0.005,
        explore_generations=0,
        min_hamming=2,
    )
    b.reset()
    stopped = False
    for g in range(10):
        stopped = b.should_stop(g, 1.0, 0.0, diversity=2.0)  # 多样性坍塌到 min_hamming
        if stopped:
            break
    assert stopped
    assert b.stop_reason() == STOP_CONVERGED


def test_improvement_resets_patience() -> None:
    b = Budget(
        profile="quick",
        wall_clock_hours=1000.0,
        max_generations=1000,
        patience=3,
        min_delta=0.005,
        explore_generations=0,
    )
    b.reset()
    assert b.should_stop(0, 1.0, 0.0) is False
    assert b.should_stop(1, 1.0, 0.0) is False
    assert b.should_stop(2, 2.0, 0.0) is False  # 大幅改进 → 重置
    assert b.generations_since_improvement(2) == 0
    assert b.should_stop(3, 2.0, 0.0) is False
    assert b.generations_since_improvement(3) == 1


# ── 「quick 轨迹是 standard 前缀」的可复现性 ────────────────────────────────


class _ProximityEvaluator:
    """适应度 = 与目标公式的负汉明距离（制造强收敛压力用于测试）。"""

    def __init__(self, target: np.ndarray) -> None:
        self.target = np.asarray(target, dtype=np.int64)

    def evaluate(self, tokens: np.ndarray) -> EvalResult:
        d = int(np.count_nonzero(np.asarray(tokens) != self.target))
        return EvalResult(train_score=-float(d), val_score=-float(d), status="ok")


def _budget(max_generations: int) -> Budget:
    return Budget(
        profile="quick",
        wall_clock_hours=1000.0,
        max_generations=max_generations,
        pop_size=32,
        elite_size=6,
        tournament_k=3,
        patience=1000,
        min_delta=0.005,
        explore_generations=2,
        immigrant_trigger=5,
        immigrant_ratio=0.2,
        island_count=2,
        island_migrate_every=3,
        island_migrate_top=2,
        min_hamming=2,
    )


def test_shorter_run_is_prefix_of_longer_run() -> None:
    cfg = GASearchConfig()
    target = random_feasible(np.random.default_rng(99), 8)
    ev = _ProximityEvaluator(target)

    ga_short = RpnGA(cfg, seed=7)
    ga_short.run(_budget(8), ev)
    ga_long = RpnGA(cfg, seed=7)
    ga_long.run(_budget(20), ev)

    short_best = [s.best_fitness for s in ga_short.last_result.history]  # type: ignore[union-attr]
    long_best = [s.best_fitness for s in ga_long.last_result.history]  # type: ignore[union-attr]
    assert len(short_best) < len(long_best)
    assert short_best == long_best[: len(short_best)], "同种子 + 同参数：短跑轨迹应为长跑前缀"
