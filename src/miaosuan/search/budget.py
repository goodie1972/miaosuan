# -*- coding: utf-8 -*-
# Copyright (c) 2026 Original Author
# Copyright (c) 2026 Your Company
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""早停与三档预算（架构 §6.2，M11）。

**原则（不可协商，A12 2026-09-10）**：**墙钟为硬约束、代数预算为辅**；每次运行显式指定档位，
默认 ``standard``。停止原因写入 :attr:`Budget.stop_reason`，取值：

``WALL_CLOCK`` | ``EARLY_STOP`` | ``MAX_GENERATIONS`` | ``CONVERGED`` | ``RUNNING``

三档参数严格取自架构 §6.2：

===========  ==========  ==============  ==========
参数          quick       standard(默认)   deep
===========  ==========  ==============  ==========
墙钟(h)        0.5         2.0             8.0
代数上限        60          300             1200
种群 P          96          256             384
精英            12          32              48
锦标赛 k        3           3               4
patience       15          40              80
min_delta      0.005       0.005           0.003
探索期          20          100             300
移民触发        20          50              80
移民比例        0.20        0.30            0.30
岛屿数          1(关)       4               4
迁移间隔        —           100             250
===========  ==========  ==============  ==========

``min_hamming``（多样性硬约束）三档一致 = 2（治 R2，不随预算缩放）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..errors import ConfigError

__all__ = [
    "DEFAULT_PROFILE",
    "PROFILE_NAMES",
    "STOP_EARLY_STOP",
    "STOP_MAX_GENERATIONS",
    "STOP_RUNNING",
    "STOP_WALL_CLOCK",
    "STOP_CONVERGED",
    "Budget",
]

STOP_RUNNING = "RUNNING"
STOP_WALL_CLOCK = "WALL_CLOCK"
STOP_EARLY_STOP = "EARLY_STOP"
STOP_MAX_GENERATIONS = "MAX_GENERATIONS"
STOP_CONVERGED = "CONVERGED"

DEFAULT_PROFILE = "standard"
PROFILE_NAMES: tuple[str, ...] = ("quick", "standard", "deep")

#: §6.2 三档参数表（键名 → (quick, standard, deep)）
_PROFILE_TABLE: dict[str, tuple[float, float, float]] = {
    "wall_clock_hours": (0.5, 2.0, 8.0),
    "max_generations": (60, 300, 1200),
    "pop_size": (96, 256, 384),
    "elite_size": (12, 32, 48),
    "tournament_k": (3, 3, 4),
    "patience": (15, 40, 80),
    "min_delta": (0.005, 0.005, 0.003),
    "explore_generations": (20, 100, 300),
    "immigrant_trigger": (20, 50, 80),
    "immigrant_ratio": (0.20, 0.30, 0.30),
    "island_count": (1, 4, 4),
    "island_migrate_every": (1, 100, 250),  # quick 关岛屿，间隔无意义
    "island_migrate_top": (0, 5, 5),
}

_INT_FIELDS = frozenset(
    {
        "max_generations",
        "pop_size",
        "elite_size",
        "tournament_k",
        "patience",
        "explore_generations",
        "immigrant_trigger",
        "island_count",
        "island_migrate_every",
        "island_migrate_top",
    }
)


def _as_type(name: str, value: float) -> float | int:
    return int(round(value)) if name in _INT_FIELDS else float(value)


@dataclass
class Budget:
    """运行期预算 + 早停状态机（可复现）。

    :param profile: 档位名（``quick`` / ``standard`` / ``deep``）。
    :param wall_clock_hours: 墙钟硬预算（秒级判定）。
    :param max_generations: 代数兜底上限（软）。
    :param pop_size: 种群规模 P。
    :param elite_size: 精英数。
    :param tournament_k: 锦标赛规模 k。
    :param patience: 精修期连续无改进多少代触发早停（决策机会数）。
    :param min_delta: 判定「有改进」的最小增益。
    :param explore_generations: 探索期代数（此期间不早停）。
    :param immigrant_trigger: 连续无改进多少代注入移民。
    :param immigrant_ratio: 移民比例。
    :param island_count: 岛屿数（1 = 关闭）。
    :param island_migrate_every: 迁移间隔（代）。
    :param island_migrate_top: 每次迁移的 top-k。
    :param min_hamming: 多样性汉明距离硬下界（三档一致 = 2）。
    """

    profile: str = DEFAULT_PROFILE
    wall_clock_hours: float = 2.0
    max_generations: int = 300
    pop_size: int = 256
    elite_size: int = 32
    tournament_k: int = 3
    patience: int = 40
    min_delta: float = 0.005
    explore_generations: int = 100
    immigrant_trigger: int = 50
    immigrant_ratio: float = 0.30
    island_count: int = 4
    island_migrate_every: int = 100
    island_migrate_top: int = 5
    min_hamming: int = 2

    # 运行期状态（不参与构造，不参与相等性）
    _reason: str = field(default=STOP_RUNNING, init=False, repr=False)
    _best_seen: float = field(default=float("-inf"), init=False, repr=False)
    _last_improve_gen: int = field(default=0, init=False, repr=False)
    _last_change_gen: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.wall_clock_hours <= 0:
            raise ConfigError("wall_clock_hours 必须为正")
        if self.max_generations < 1:
            raise ConfigError("max_generations 必须 >= 1")
        if self.elite_size > self.pop_size:
            raise ConfigError("elite_size 不得大于 pop_size")
        if self.min_hamming < 0:
            raise ConfigError("min_hamming 不得为负")
        if self.island_count < 1:
            raise ConfigError("island_count 必须 >= 1")

    # ── 构造 ────────────────────────────────────────────────────────────────

    @classmethod
    def from_profile(cls, name: str = DEFAULT_PROFILE) -> Budget:
        """按档位名构造预算（§6.2 参数表）。

        :raises ConfigError: 未知档位名。
        """
        if name not in PROFILE_NAMES:
            raise ConfigError(
                f"未知预算档位 {name!r}（可选 {PROFILE_NAMES}）", context={"profile": name}
            )
        idx = PROFILE_NAMES.index(name)
        kwargs: dict[str, object] = {"profile": name}
        for key, values in _PROFILE_TABLE.items():
            kwargs[key] = _as_type(key, values[idx])
        return cls(**kwargs)  # type: ignore[arg-type]

    # ── 状态机 ──────────────────────────────────────────────────────────────

    @property
    def wall_clock_seconds(self) -> float:
        """墙钟硬预算（秒）。"""
        return self.wall_clock_hours * 3600.0

    @property
    def islands_enabled(self) -> bool:
        """是否启用岛屿模型（``island_count > 1``）。"""
        return self.island_count > 1

    def reset(self) -> None:
        """重置早停状态（复用于多次运行）。"""
        self._reason = STOP_RUNNING
        self._best_seen = float("-inf")
        self._last_improve_gen = 0
        self._last_change_gen = 0

    def should_stop(
        self,
        generation: int,
        best: float,
        elapsed_seconds: float,
        *,
        diversity: float | None = None,
    ) -> bool:
        """判断是否停止（并记录 ``stop_reason``）。

        判定顺序：**墙钟 → 代数上限 → 早停/收敛**（墙钟永远优先）。
        早停仅在探索期之后生效；精修期内连续 ``patience`` 代最优值提升 < ``min_delta`` 即停。

        :param generation: 当前已完成的代数（0 基）。
        :param best: 当前最优适配度。
        :param elapsed_seconds: 已耗墙钟（秒）。
        :param diversity: 当前种群多样性（平均两两汉明距离），用于区分 ``CONVERGED``。
        """
        self.note_improvement(generation, best)

        if elapsed_seconds >= self.wall_clock_seconds:
            self._reason = STOP_WALL_CLOCK
            return True
        if generation + 1 >= self.max_generations:
            self._reason = STOP_MAX_GENERATIONS
            return True
        if generation >= self.explore_generations:
            stall = int(generation) - self._last_improve_gen
            if stall >= self.patience:
                no_real_change = (int(generation) - self._last_change_gen) >= self.patience
                collapsed = diversity is not None and diversity <= float(self.min_hamming)
                self._reason = STOP_CONVERGED if (no_real_change and collapsed) else STOP_EARLY_STOP
                return True
        self._reason = STOP_RUNNING
        return False

    def stop_reason(self) -> str:
        """返回最近一次判定/停止的原因。"""
        return self._reason

    def note_improvement(self, generation: int, best: float) -> None:
        """登记一次适配度观测，更新「改进 / 变化」时间戳（供早停与移民判定）。"""
        if not (best > float("-inf")):
            return
        if best > self._best_seen + self.min_delta:
            self._best_seen = float(best)
            self._last_improve_gen = int(generation)
            self._last_change_gen = int(generation)
        elif best > self._best_seen:
            self._best_seen = float(best)
            self._last_change_gen = int(generation)

    def generations_since_improvement(self, generation: int) -> int:
        """自上次「有改进」以来经过的代数（供移民触发判定）。"""
        return int(generation) - self._last_improve_gen
