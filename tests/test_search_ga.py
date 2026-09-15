# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""RpnGA 引擎 + AM 口径适应度单测（M10/M11）。"""

from __future__ import annotations

import numpy as np
import pytest

from miaosuan.config import GASearchConfig
from miaosuan.core.features import compute_features
from miaosuan.data.panel import PANEL_FIELDS, Panel
from miaosuan.search.budget import STOP_MAX_GENERATIONS, Budget
from miaosuan.search.ga import (
    AMFitnessEvaluator,
    EvalResult,
    RpnGA,
    _apply_ic_gate,
    _repetition_penalty,
    build_walk_forward_folds,
)
from miaosuan.search.rpn import Individual, is_feasible, random_feasible

# ── AM 口径辅助函数 ──────────────────────────────────────────────────────────


def test_build_walk_forward_folds_rolling() -> None:
    folds = build_walk_forward_folds(8000, n_folds=5, gap=20)
    assert len(folds) == 4
    for f in folds:
        ordered = f["train_start"] < f["train_end"] <= f["val_start"] < f["val_end"]
        assert ordered
        assert f["val_start"] == f["train_end"] + f["gap"]
    # AM 等价：total_required > T 时把 gap 收敛到 ``(T - fold_size*n)//n``；
    # 由于 ``fold_size = T // n``，典型 T 下该值为 0（冻结 AM 已如此，逐点保留）。
    assert folds[0]["gap"] == 0
    # 退化：数据过短
    degenerate = build_walk_forward_folds(3, n_folds=5, gap=20)
    assert degenerate == [{"train_start": 0, "train_end": 3, "val_start": 0, "val_end": 3, "gap": 0}]


def test_apply_ic_gate_directions() -> None:
    assert _apply_ic_gate(1.0, 0.05) == pytest.approx(1.15)
    assert _apply_ic_gate(1.0, -0.05) == pytest.approx(0.75)
    assert _apply_ic_gate(1.0, 0.001) == pytest.approx(1.0)


def test_repetition_penalty() -> None:
    assert _repetition_penalty(np.array([1, 2, 3, 4])) == 0.0
    assert _repetition_penalty(np.array([7, 7, 7, 7])) == pytest.approx(0.9)
    assert _repetition_penalty(np.array([7, 7, 8, 9])) == pytest.approx(0.3)


# ── 合成评估器（快、确定、强收敛压力）──────────────────────────────────────


class ProximityEvaluator:
    """适应度 = 与目标公式的负汉明距离。"""

    def __init__(self, target: np.ndarray) -> None:
        self.target = np.asarray(target, dtype=np.int64)
        self.calls = 0

    def evaluate(self, tokens: np.ndarray) -> EvalResult:
        self.calls += 1
        d = int(np.count_nonzero(np.asarray(tokens) != self.target))
        return EvalResult(train_score=-float(d), val_score=-float(d), status="ok")


def _tiny_budget(max_generations: int = 12, *, island_count: int = 2) -> Budget:
    return Budget(
        profile="quick",
        wall_clock_hours=1000.0,
        max_generations=max_generations,
        pop_size=40,
        elite_size=6,
        tournament_k=3,
        patience=1000,
        min_delta=0.005,
        explore_generations=2,
        immigrant_trigger=3,
        immigrant_ratio=0.2,
        island_count=island_count,
        island_migrate_every=3,
        island_migrate_top=2,
        min_hamming=2,
    )


def test_ga_runs_and_returns_feasible_best() -> None:
    cfg = GASearchConfig()
    target = random_feasible(np.random.default_rng(11), 8)
    ga = RpnGA(cfg, seed=3)
    best = ga.run(_tiny_budget(), ProximityEvaluator(target))
    assert isinstance(best, Individual)
    assert is_feasible(best.tokens)
    assert best.fitness > float("-inf")
    result = ga.last_result
    assert result is not None
    assert result.stop_reason == STOP_MAX_GENERATIONS
    assert result.generations >= 1


def test_ga_improves_over_initial() -> None:
    cfg = GASearchConfig()
    target = random_feasible(np.random.default_rng(12), 8)
    ga = RpnGA(cfg, seed=5)
    ga.run(_tiny_budget(max_generations=15), ProximityEvaluator(target))
    hist = ga.last_result.history  # type: ignore[union-attr]
    assert hist[-1].best_fitness >= hist[0].best_fitness


def test_ga_deterministic_same_seed() -> None:
    cfg = GASearchConfig()
    target = random_feasible(np.random.default_rng(13), 8)
    ga1 = RpnGA(cfg, seed=42)
    b1 = ga1.run(_tiny_budget(), ProximityEvaluator(target))
    ga2 = RpnGA(cfg, seed=42)
    b2 = ga2.run(_tiny_budget(), ProximityEvaluator(target))
    assert b1.key() == b2.key()
    assert b1.fitness == b2.fitness


def test_ga_diversity_never_below_40pct_of_initial() -> None:
    cfg = GASearchConfig()
    target = random_feasible(np.random.default_rng(14), 8)
    ga = RpnGA(cfg, seed=8)
    ga.run(_tiny_budget(max_generations=25), ProximityEvaluator(target))
    result = ga.last_result
    assert result is not None
    assert result.diversity_ratio >= 0.4, (
        f"多样性最低点 {result.min_diversity:.3f} 低于初始 {result.initial_diversity:.3f} 的 40%"
    )
    # 逐代证据也须满足
    for s in result.history:
        assert s.diversity >= 0.4 * result.initial_diversity


def test_ga_min_hamming_hard_constraint_within_island() -> None:
    cfg = GASearchConfig()
    target = random_feasible(np.random.default_rng(15), 8)
    ga = RpnGA(cfg, seed=9)
    ga.run(_tiny_budget(max_generations=10, island_count=1), ProximityEvaluator(target))
    pop = ga.final_populations[0]
    keys = [ind.key() for ind in pop]
    assert len(set(keys)) == len(keys), "同代不得出现完全相同的个体（min_hamming ≥ 2 至少排除重复）"
    from miaosuan.search.rpn import hamming

    for i in range(len(pop)):
        for j in range(i + 1, len(pop)):
            assert hamming(pop[i].tokens, pop[j].tokens) >= 2


def test_ga_evaluation_cache_dedup() -> None:
    cfg = GASearchConfig()
    ga = RpnGA(cfg, seed=1)
    ev = ProximityEvaluator(random_feasible(np.random.default_rng(0), 8))
    ind = Individual(tokens=random_feasible(np.random.default_rng(2), 8))
    ga._evaluate(ind, ev)
    ga._evaluate(ind, ev)
    assert ev.calls == 1
    assert ga.n_evaluations == 1


def test_ga_migration_does_not_add_evaluations() -> None:
    # 迁移为纯数据搬运：开启岛屿不应显著增加评估次数（每代 ≤ 种群规模）。
    cfg = GASearchConfig()
    target = random_feasible(np.random.default_rng(16), 8)
    ga = RpnGA(cfg, seed=11)
    ga.run(_tiny_budget(max_generations=10, island_count=4), ProximityEvaluator(target))
    result = ga.last_result
    assert result is not None
    # 评估次数上界：初始化 + 每代 ≤ 种群规模（+ 多样性地板兜底注入）
    upper = 40 * (result.generations + 1) + result.n_diversity_immigrants + 40
    assert result.n_evaluations <= upper


def test_ga_top_individuals_dedup_sorted() -> None:
    cfg = GASearchConfig()
    target = random_feasible(np.random.default_rng(17), 8)
    ga = RpnGA(cfg, seed=13)
    ga.run(_tiny_budget(max_generations=10), ProximityEvaluator(target))
    top = ga.top_individuals(5)
    assert 1 <= len(top) <= 5
    fitnesses = [ind.fitness for ind in top]
    assert fitnesses == sorted(fitnesses, reverse=True)
    assert len({ind.key() for ind in top}) == len(top)


# ── AMFitnessEvaluator 一致性（合成数据，无需外部文件）───────────────────────


def _synthetic_panel(n: int = 1600, seed: int = 0) -> Panel:
    rng = np.random.default_rng(seed)
    price = 100.0 * np.cumprod(1.0 + rng.normal(0, 0.002, (1, n)), axis=1)
    fields = {
        "open": price.astype(np.float32),
        "high": (price * 1.001).astype(np.float32),
        "low": (price * 0.999).astype(np.float32),
        "close": (price * 1.0005).astype(np.float32),
        "volume": rng.uniform(100, 200, (1, n)).astype(np.float32),
    }
    assert set(fields) == set(PANEL_FIELDS)
    time = (np.arange(n, dtype=np.int64) + 1) * 3600
    return Panel.from_arrays(
        fields, time, symbols=("XAUUSD",), timeframe="H1", market_profile_name="FOREX_XAUUSD"
    )


def test_am_fitness_evaluator_smoke() -> None:
    panel = _synthetic_panel(1600)
    feats = compute_features(panel.to_raw_dict())
    from miaosuan.search.mine import compute_target_ret

    tret = compute_target_ret(panel.open)
    ev = AMFitnessEvaluator(feats, tret, cost_rate=0.0003, periods_per_year=6240, n_folds=5, gap=20)
    assert len(ev.folds) >= 1
    res = ev.evaluate(random_feasible(np.random.default_rng(4), 8))
    assert res.status in {"ok", "none", "const"}
    assert np.isfinite(res.val_score)
    assert ev.n_evaluated == 1


def test_am_fitness_evaluator_none_for_infeasible() -> None:
    panel = _synthetic_panel(800)
    feats = compute_features(panel.to_raw_dict())
    from miaosuan.search.mine import compute_target_ret

    tret = compute_target_ret(panel.open)
    ev = AMFitnessEvaluator(feats, tret, cost_rate=0.0003, periods_per_year=6240)
    # 特征越界 token（>= F）→ 不可求值
    bad = np.full(8, 9999, dtype=np.int64)
    res = ev.evaluate(bad)
    assert res.status == "none"
    assert res.val_score == -5.0
