"""单一配置源（架构 §9.2）。

设计目标：

* **一个配置对象树**（:class:`AppConfig`），而非 AM 的「根目录 config + model_core/config」
  双配置架构倒置（``engine.py:46-51`` 反向 import 根目录 config）；
* **依赖注入**：各层（``core`` / ``search`` / ``gate`` / ``adapters`` …）通过函数/构造
  参数接收所需配置片段，**从不**自行去读全局或环境变量；
* **import 时零副作用**：本模块在 import 时不读取任何环境变量、不做文件 IO。
  需要从环境构造配置时，显式调用 :meth:`AppConfig.from_env`（由 CLI 边界触发）。

一切可复现相关参数（seed、窗口、成本、切分比例）都可经 :meth:`AppConfig.to_snapshot`
导出，写入产物 ``provenance.config_snapshot``（架构 §3.1、§9.2）。
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from .errors import ConfigError

__all__ = [
    "ConfigError",  # 便于统一从 config 导入错误类型
    "GASearchConfig",
    "BudgetConfig",
    "SplitConfig",
    "CostConfig",
    "LoggingConfig",
    "AppConfig",
]

# ── 子配置（均为 frozen dataclass，不可变、可安全共享）─────────────────────


@dataclass(frozen=True)
class GASearchConfig:
    """RPN-GA 搜索器配置（架构 §1.3）。

    :param formula_len: 定长公式 token 数（Q6-A 保持 8）。
    :param pop_size: 种群规模（对齐 AM ``BATCH_SIZE=192`` 量级）。
    :param elite_size: 精英池规模。
    :param tournament_k: 锦标赛选择规模。
    :param min_hamming: 多样性硬约束：新个体与精英库最小汉明距离下界（治 R2 熵坍塌）。
    :param p_mutate_point: 单点替换变异概率。
    :param p_mutate_subtree: 子树增删变异概率。
    :param p_crossover: 均匀交叉概率。
    :param n_islands: 岛屿数。
    :param island_migrate_every: 每隔多少代做一次岛屿迁移。
    :param island_migrate_top: 每次迁移的 top-k 个体数。
    :param immigrant_ratio: 连续无改进达到阈值时注入的随机个体比例。
    """

    formula_len: int = 8
    pop_size: int = 256
    elite_size: int = 32
    tournament_k: int = 3
    min_hamming: int = 2
    p_mutate_point: float = 0.5
    p_mutate_subtree: float = 0.2
    p_crossover: float = 0.3
    n_islands: int = 4
    island_migrate_every: int = 100
    island_migrate_top: int = 5
    immigrant_ratio: float = 0.30

    def __post_init__(self) -> None:
        if self.formula_len < 1:
            raise ConfigError("formula_len 必须 >= 1", context={"formula_len": self.formula_len})
        if self.elite_size > self.pop_size:
            raise ConfigError(
                "elite_size 不得大于 pop_size",
                context={"elite_size": self.elite_size, "pop_size": self.pop_size},
            )
        if not 0.0 <= self.p_mutate_point <= 1.0:
            raise ConfigError("p_mutate_point 必须在 [0,1]")

    @property
    def elite_ratio(self) -> float:
        """精英占比。"""
        return self.elite_size / self.pop_size


@dataclass(frozen=True)
class BudgetConfig:
    """早停与预算（治 R1，架构 §6.2）。

    :param profile: 预算档位名（quick / standard / deep）。
    :param wall_clock_hours: **墙钟硬预算**（真正的硬约束）。
    :param max_generations: 代数兜底上限。
    :param patience: 连续多少代无改进触发早停。
    :param min_delta: 判定「有改进」的最小增益。
    :param explore_generations: 探索期代数（此期间不早停）。
    """

    profile: str = "standard"
    wall_clock_hours: float = 2.0
    max_generations: int = 300
    patience: int = 40
    min_delta: float = 0.005
    explore_generations: int = 100

    def __post_init__(self) -> None:
        if self.wall_clock_hours <= 0:
            raise ConfigError("wall_clock_hours 必须为正")
        if self.max_generations < 1:
            raise ConfigError("max_generations 必须 >= 1")

    # 预置档位（架构 §10 A12 建议）
    @classmethod
    def quick(cls) -> BudgetConfig:
        return cls(profile="quick", wall_clock_hours=0.5, max_generations=80, patience=25)

    @classmethod
    def standard(cls) -> BudgetConfig:
        return cls(profile="standard", wall_clock_hours=2.0, max_generations=300, patience=40)

    @classmethod
    def deep(cls) -> BudgetConfig:
        return cls(profile="deep", wall_clock_hours=8.0, max_generations=1200, patience=60)


@dataclass(frozen=True)
class SplitConfig:
    """三段切分 + purge/embargo（架构 §6.1）。

    :param train_ratio: 训练段比例。
    :param val_ratio: 验证段比例。
    :param holdout_ratio: hold-out 段比例（永不参与训练/选择）。
    :param purge_gap: 段间 purge 间隔（默认对齐 AM ``WF_GAP=20``）。
    :param embargo: 防标签重叠的 embargo。
    :param n_wf_folds: walk-forward 折数。
    """

    train_ratio: float = 0.6
    val_ratio: float = 0.2
    holdout_ratio: float = 0.2
    purge_gap: int = 20
    embargo: int = 4
    n_wf_folds: int = 4

    def __post_init__(self) -> None:
        total = self.train_ratio + self.val_ratio + self.holdout_ratio
        if abs(total - 1.0) > 1e-9:
            raise ConfigError("三段比例之和必须为 1.0", context={"total": total})
        if self.n_wf_folds < 1:
            raise ConfigError("n_wf_folds 必须 >= 1")


@dataclass(frozen=True)
class CostConfig:
    """交易成本画像（对齐 MarketProfile，架构 §3.1）。

    :param cost_profile: 成本画像名（如 ``FOREX_XAUUSD``）。
    :param cost_rate: 单边成本率（手续费 + 滑点）。
    :param sensitivity: 成本敏感性档位倍率（架构 §6.4：0.5x/1x/2x/3x）。
    """

    cost_profile: str = "FOREX_XAUUSD"
    cost_rate: float = 0.0002
    sensitivity: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0)

    def __post_init__(self) -> None:
        if self.cost_rate < 0:
            raise ConfigError("cost_rate 不得为负")


@dataclass(frozen=True)
class LoggingConfig:
    """结构化日志配置（架构 §9.5）。

    :param level: 日志级别名（DEBUG/INFO/...）。
    :param json_lines: 是否输出 JSON 行（生产 True；测试可关）。
    """

    level: str = "INFO"
    json_lines: bool = True


# ── 顶层配置树 ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AppConfig:
    """妙算顶层配置树（单一配置源）。

    通过 :meth:`from_mapping` / :meth:`with_overrides` 构造与派生；
    各层只接收自己关心的子配置，绝不反向 import 本模块读取全局状态。
    """

    #: 全局随机种子（写入 provenance，禁止隐式全局随机，架构 §9.6）
    seed: int = 20260910
    #: 目标市场/标的/周期
    symbol: str = "XAUUSD"
    timeframe: str = "H1"
    search: GASearchConfig = field(default_factory=GASearchConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # ── 派生 ───────────────────────────────────────────────────────────────

    def with_overrides(self, **kwargs: Any) -> AppConfig:
        """基于当前配置派生新配置（frozen dataclass 的不可变替换）。"""
        return replace(self, **kwargs)

    def to_snapshot(self) -> dict[str, Any]:
        """导出可 JSON 序列化的配置快照，写入 ``provenance.config_snapshot``。"""
        snapshot: dict[str, Any] = _to_jsonable(dataclasses.asdict(self))
        return snapshot

    # ── 从映射 / 环境构造（显式调用，import 时不触发）───────────────────────

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> AppConfig:
        """从（可能是 YAML 解析出的）嵌套映射构造配置。

        未知键抛出 :class:`ConfigError`，避免静默吞掉拼写错误。
        """
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ConfigError(f"未知配置键: {sorted(unknown)}")

        sub_types: dict[str, type] = {
            "search": GASearchConfig,
            "budget": BudgetConfig,
            "split": SplitConfig,
            "cost": CostConfig,
            "logging": LoggingConfig,
        }
        kwargs: dict[str, Any] = {}
        for key, value in data.items():
            if key in sub_types and isinstance(value, Mapping):
                kwargs[key] = _build_sub(sub_types[key], value)
            else:
                kwargs[key] = value
        return cls(**kwargs)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> AppConfig:
        """从环境变量构造**最小**配置（仅覆盖 symbol/timeframe/seed/日志级别）。

        这是妙算唯一允许读取环境变量的入口，且必须被显式调用（CLI 边界）。
        ``core/`` 永远不会调用它 —— 这是依赖方向铁律的一部分。
        """
        source = os.environ if env is None else env
        base = cls()
        overrides: dict[str, Any] = {}
        if sym := source.get("MIAOSUAN_SYMBOL"):
            overrides["symbol"] = sym
        if tf := source.get("MIAOSUAN_TIMEFRAME"):
            overrides["timeframe"] = tf
        if raw_seed := source.get("MIAOSUAN_SEED"):
            try:
                overrides["seed"] = int(raw_seed)
            except ValueError as exc:  # pragma: no cover - 防御性
                raise ConfigError("MIAOSUAN_SEED 必须为整数", context={"value": raw_seed}) from exc
        if lvl := source.get("MIAOSUAN_LOG_LEVEL"):
            overrides["logging"] = replace(base.logging, level=lvl.upper())
        return base.with_overrides(**overrides)


# ── 内部工具 ───────────────────────────────────────────────────────────────


def _build_sub(cls: type, value: Mapping[str, Any]) -> Any:
    """用映射构造子配置，未知键报错。"""
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(value) - known
    if unknown:
        raise ConfigError(f"{cls.__name__} 未知配置键: {sorted(unknown)}")
    return cls(**dict(value))


def _to_jsonable(obj: Any) -> Any:
    """递归把 dataclass/asdict 结果转换为 JSON 友好类型（tuple -> list）。"""
    if isinstance(obj, Mapping):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj
