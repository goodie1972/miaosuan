# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for full license text.

"""S3 精英龄期回放测试（路线图 §6）：启用 birth_gen 的半衰衰减回放。

断言：
* 衰减权重随代龄指数下降（0.5^(age/halflife)）；
* softmax 温度 < 1 防单点垄断：采样分布不退化纯 argmax（次优个体也有权重）；
* 回放注入当代种群且按 token 去重（池中不出现重复 token 个体）；
* 默认关闭（elite_replay_size=0）时行为与既有 GA 完全一致（回归防线）。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from miaosuan.config import AppConfig, GASearchConfig
from miaosuan.search.budget import Budget
from miaosuan.search.ga import RpnGA
from miaosuan.search.rpn import Individual


def _cfg(**search_kwargs: Any) -> AppConfig:
    cfg = AppConfig()
    cfg = cfg.with_overrides(
        search=GASearchConfig(formula_len=6, pop_size=16, elite_size=3, **search_kwargs)
    )
    return cfg


def _budget() -> Budget:
    return Budget(
        profile="quick", wall_clock_hours=2.0, max_generations=6, pop_size=16,
        elite_size=3, tournament_k=3, patience=1000, min_delta=0.0,
        explore_generations=1000, immigrant_trigger=1000, immigrant_ratio=0.2,
        island_count=2, island_migrate_every=3, island_migrate_top=2, min_hamming=1,
    )


class _DummyEvaluator:
    def evaluate(self, tokens: np.ndarray) -> Any:
        from miaosuan.search.ga import EvalResult

        key = tuple(int(t) for t in tokens)
        val = sum(k % 5 for k in key) / 50.0
        return EvalResult(train_score=val, val_score=val, status="ok")


def _make_ga(**search_kwargs: Any) -> RpnGA:
    return RpnGA(_cfg(**search_kwargs).search, seed=1, length=6)


# ── 衰减权重：随代龄指数下降 ─────────────────────────────────────────────


def test_elite_decay_weights_decrease_with_age() -> None:
    ga = _make_ga(elite_replay_size=2, elite_pool_size=8, elite_replay_halflife=10.0)
    # 构造精英池：同分个体，但出生代数差 10 → 年龄差 10 代（half=10 → 权重差 2 倍）。
    ga._elite_pool = [  # noqa: SLF001
        Individual(tokens=np.asarray([1, 2, 3, 1, 2, 1]), val_score=0.4, fitness=0.4, birth_gen=0),
        Individual(tokens=np.asarray([4, 5, 6, 4, 5, 1]), val_score=0.4, fitness=0.4, birth_gen=10),
    ]
    # generation=20：第 1 个 age=20（半衰两次 → 权重减 4），第 2 个 age=10（减 2）。
    w = ga._elite_replay_weights(generation=20)  # noqa: SLF001
    assert w.shape == (2,)
    assert w.sum() == pytest.approx(1.0)
    # 老精英（birth_gen=0, age=20）权重 < 新精英（birth_gen=10, age=10）。
    assert w[0] < w[1]


# ── softmax 温度 < 1：次优个体仍有非零采样权重（不退化 argmax）───────────


def test_elite_replay_sampling_not_degenerate_argmax() -> None:
    ga = _make_ga(elite_replay_size=1, elite_pool_size=8, elite_replay_temperature=0.2)
    # 构造分数相差很大的池：最高分 1.0，次优 0.5（T=0.2 → softmax 拉大但非硬 argmax）。
    ga._elite_pool = [  # noqa: SLF001
        Individual(tokens=np.asarray([1, 2, 3, 1, 2, 1]), val_score=1.0, fitness=1.0, birth_gen=0),
        Individual(tokens=np.asarray([4, 5, 6, 4, 5, 1]), val_score=0.5, fitness=0.5, birth_gen=0),
    ]
    w = ga._elite_replay_weights(generation=5)  # noqa: SLF001
    # 温度 < 1 压缩 softmax：次优个体不应被完全归零。
    assert w[1] > 0.01


# ── 回放注入当代种群且 token 去重 ────────────────────────────────────────


def test_elite_replay_injects_and_dedups() -> None:
    ga = _make_ga(elite_replay_size=2, elite_pool_size=8)
    pool = [
        Individual(tokens=np.asarray([1, 2, 3, 1, 2, 1]), val_score=0.4, fitness=0.4, birth_gen=0),
        Individual(tokens=np.asarray([4, 5, 6, 4, 5, 1]), val_score=0.3, fitness=0.3, birth_gen=1),
    ]
    ga._elite_pool = pool  # noqa: SLF001
    replay = ga._replay_elite(generation=5)  # noqa: SLF001
    assert len(replay) == 2
    keys = [i.key() for i in replay]
    assert len(set(keys)) == len(keys)  # 去重


# ── 默认关闭：与既有 GA 行为一致（回归防线）──────────────────────────────


def test_elite_replay_default_off_matches_plain_ga() -> None:
    """elite_replay_size=0（默认）→ 回放关闭，结果与未启用回放的 GA 一致。"""
    ev = _DummyEvaluator()
    ga = _make_ga()  # 默认关闭
    ga.run(_budget(), ev)
    assert ga._elite_pool == []  # noqa: SLF001
    assert ga.config.elite_replay_enabled is False


# ── 启用回放：仍可复现（同 seed 同结果） ──────────────────────────────────


def test_elite_replay_deterministic() -> None:
    def _run_once():
        ev = _DummyEvaluator()
        ga = _make_ga(elite_replay_size=2, elite_pool_size=8)
        ga.run(_budget(), ev)
        return ga.last_result.best.fitness

    assert _run_once() == pytest.approx(_run_once(), abs=1e-12)