"""岛屿模型（架构 §1.3、M10）。

* **分岛**：把总种群 ``pop_size`` 均分到 ``island_count`` 个岛（余数分配到前若干岛）；
  每岛独立演化（选择 / 变异 / 交叉 / 多样性约束）。
* **迁移**：环形拓扑，每 ``migrate_every`` 代把各岛 top-``migrate_top`` 个体迁往下一岛，
  替换目标岛的最差个体。**迁移是纯数据搬运 —— 被迁移个体的适配度已评估，绝不重复评估**
  （对齐架构「岛屿不增加评估次数」的约束；规避 AM 在 CPU 下串行跑 N 倍的坑）。
* ``quick`` 档 ``island_count=1`` → 关闭岛屿（:meth:`IslandModel.enabled` 为 False）。

多样性度量 = **平均两两汉明距离**（跨全部岛），用于「无熵坍塌」的可机器验证证据。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from .rpn import Individual, hamming

__all__ = ["IslandModel", "partition_population"]


def partition_population(population: list[Individual], n_islands: int) -> list[list[Individual]]:
    """把 ``population`` 均分为 ``n_islands`` 个岛（余数分配到前若干岛）。"""
    n = max(1, int(n_islands))
    if n == 1:
        return [list(population)]
    size = len(population)
    base = size // n
    extra = size % n
    islands: list[list[Individual]] = []
    start = 0
    for i in range(n):
        take = base + (1 if i < extra else 0)
        islands.append(list(population[start : start + take]))
        start += take
    return islands


@dataclass
class IslandModel:
    """岛屿模型（分岛 + 环形迁移）。

    :param n_islands: 岛屿数（1 = 关闭）。
    :param migrate_every: 迁移间隔（代）。
    :param migrate_top: 每次迁移的 top-k。
    """

    n_islands: int = 4
    migrate_every: int = 100
    migrate_top: int = 5

    def __post_init__(self) -> None:
        if self.n_islands < 1:
            raise ValueError("n_islands 必须 >= 1")
        if self.migrate_every < 1:
            raise ValueError("migrate_every 必须 >= 1")

    @property
    def enabled(self) -> bool:
        """是否启用岛屿（``n_islands > 1``）。"""
        return self.n_islands > 1

    def should_migrate(self, generation: int) -> bool:
        """第 ``generation`` 代是否应执行迁移。"""
        if not self.enabled or self.migrate_top <= 0:
            return False
        return generation > 0 and generation % self.migrate_every == 0

    def migrate(self, populations: list[list[Individual]]) -> int:
        """环形迁移（原地修改 ``populations``），返回迁移的个体数。

        岛 ``i`` 的 top-k 迁往岛 ``(i+1) % n``，替换其最差个体；已存在的同名个体不重复迁入。
        """
        n = len(populations)
        if n < 2 or self.migrate_top <= 0:
            return 0
        emigrants: list[list[Individual]] = [
            [
                copy.deepcopy(ind)
                for ind in sorted(pop, key=lambda x: x.fitness, reverse=True)[: self.migrate_top]
            ]
            for pop in populations
        ]
        moved = 0
        for i, pop in enumerate(populations):
            incoming = emigrants[(i - 1) % n]
            if not incoming:
                continue
            pop.sort(key=lambda x: x.fitness)  # 升序：最差在前
            existing = {ind.key() for ind in pop}
            for ind in incoming:
                key = ind.key()
                if key in existing:
                    continue
                if pop:
                    victim = pop[0]
                    existing.discard(victim.key())
                    pop[0] = ind
                    existing.add(key)
                    moved += 1
                    pop.sort(key=lambda x: x.fitness)
                else:  # pragma: no cover - 空岛不应出现
                    pop.append(ind)
                    existing.add(key)
                    moved += 1
        return moved

    def all_individuals(self, populations: list[list[Individual]]) -> list[Individual]:
        """展平全部岛的个体。"""
        return [ind for pop in populations for ind in pop]

    def best(self, populations: list[list[Individual]]) -> Individual:
        """跨岛最优个体。

        :raises ValueError: 全岛为空。
        """
        pool = self.all_individuals(populations)
        if not pool:
            raise ValueError("种群为空，无最优个体")
        return max(pool, key=lambda x: x.fitness)

    def diversity(self, populations: list[list[Individual]], *, max_pairs: int = 200_000) -> float:
        """平均两两汉明距离（跨全部岛）。不足 2 个个体时返回 0。"""
        pool = self.all_individuals(populations)
        m = len(pool)
        if m < 2:
            return 0.0
        n_pairs = m * (m - 1) // 2
        if n_pairs > max_pairs:
            rng = np.random.default_rng(0)
            idx_a = rng.integers(0, m, size=max_pairs)
            idx_b = rng.integers(0, m, size=max_pairs)
            total = 0
            count = 0
            for a, b in zip(idx_a, idx_b, strict=True):
                if int(a) == int(b):
                    continue
                total += hamming(pool[int(a)].tokens, pool[int(b)].tokens)
                count += 1
            return float(total / count) if count else 0.0
        total_d = 0
        for i in range(m):
            for j in range(i + 1, m):
                total_d += hamming(pool[i].tokens, pool[j].tokens)
        return float(total_d / n_pairs)
