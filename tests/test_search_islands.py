"""岛屿模型单测（M10）。"""

from __future__ import annotations

import numpy as np
import pytest

from miaosuan.search.islands import IslandModel, partition_population
from miaosuan.search.rpn import Individual, random_feasible


def _ind(seed: int, fitness: float) -> Individual:
    rng = np.random.default_rng(seed)
    return Individual(tokens=random_feasible(rng, 8), fitness=fitness, val_score=fitness, status="ok")


def test_partition_even_and_remainder() -> None:
    pop = [_ind(i, float(i)) for i in range(10)]
    parts = partition_population(pop, 4)
    assert [len(p) for p in parts] == [3, 3, 2, 2]
    assert sorted(i.fitness for p in parts for i in p) == [float(i) for i in range(10)]


def test_partition_single_island() -> None:
    pop = [_ind(i, float(i)) for i in range(5)]
    parts = partition_population(pop, 1)
    assert len(parts) == 1
    assert len(parts[0]) == 5


def test_island_model_disabled_when_single() -> None:
    model = IslandModel(n_islands=1)
    assert model.enabled is False
    assert model.should_migrate(100) is False


def test_should_migrate_on_interval() -> None:
    model = IslandModel(n_islands=4, migrate_every=10, migrate_top=2)
    assert model.should_migrate(0) is False
    assert model.should_migrate(10) is True
    assert model.should_migrate(15) is False


def test_migrate_ring_topology_and_pure_movement() -> None:
    model = IslandModel(n_islands=3, migrate_every=1, migrate_top=1)
    # 每岛一个最高分个体（fitness 递增），迁移后应出现在「下一个」岛
    islands = [
        [_ind(1, 1.0), _ind(2, 0.1)],
        [_ind(3, 2.0), _ind(4, 0.2)],
        [_ind(5, 3.0), _ind(6, 0.3)],
    ]
    top_keys = [max(pop, key=lambda x: x.fitness).key() for pop in islands]
    moved = model.migrate(islands)
    assert moved > 0
    # 岛 i 的 top 迁往岛 (i+1)%3
    for i in range(3):
        dst_keys = {ind.key() for ind in islands[(i + 1) % 3]}
        assert top_keys[i] in dst_keys


def test_migrate_copies_not_aliases() -> None:
    model = IslandModel(n_islands=2, migrate_every=1, migrate_top=1)
    islands = [[_ind(1, 5.0)], [_ind(2, 0.0)]]
    src = islands[0][0]
    model.migrate(islands)
    # 迁移副本与源对象不是同一实例（避免跨岛共享可变状态）
    assert all(ind is not src for ind in islands[1])


def test_diversity_average_pairwise_hamming() -> None:
    model = IslandModel(n_islands=1)
    identical = [Individual(tokens=np.array([0, 1, 2, 3, 4, 5, 6, 7]), fitness=1.0) for _ in range(4)]
    assert model.diversity([identical]) == 0.0

    distinct = [
        Individual(tokens=np.array([0, 0, 0, 0, 0, 0, 0, 0]), fitness=1.0),
        Individual(tokens=np.array([1, 1, 1, 1, 1, 1, 1, 1]), fitness=1.0),
    ]
    assert model.diversity([distinct]) == 8.0


def test_best_across_islands() -> None:
    model = IslandModel(n_islands=2)
    islands = [[_ind(1, 1.0)], [_ind(2, 9.0)]]
    assert model.best(islands).fitness == 9.0
    with pytest.raises(ValueError):
        model.best([[], []])


def test_invalid_construction() -> None:
    with pytest.raises(ValueError):
        IslandModel(n_islands=0)
    with pytest.raises(ValueError):
        IslandModel(n_islands=2, migrate_every=0)
