# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""统一配置适配层（向后兼容旧 config.py 接口）。

此模块保持与旧 ``config.py`` 完全相同的函数签名和行为，
但内部委托给新的统一设置系统（settings.py）。
这样可以在不修改任何现有代码的情况下，获得统一配置的所有好处：
- YAML 配置文件支持
- 环境变量覆盖 (MIAOSUAN_*)
- 热重载 (文件监听)
- 运行时覆盖
- 类型验证和错误报告

所有函数仍然是「唯一 env 边界」（架构 §9.2），即仅在此处读取环境变量。
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from .errors import ConfigError

if TYPE_CHECKING:
    from .settings import Settings


def get_config() -> Settings:
    """延迟导入统一设置系统，避免循环依赖。

    config.py 被 settings.py import 了 ConfigError，
    所以这里用函数级延迟导入打破循环。
    """
    from .settings import get_config as _get_config
    return _get_config()

__all__ = [
    "ConfigError",  # 便于统一从 config 导入错误类型
    "GASearchConfig",
    "BudgetConfig",
    "SplitConfig",
    "CostConfig",
    "LoggingConfig",
    "AppConfig",
    "DEFAULT_SHENJI_URL",
    "DEFAULT_SHENJI_TIMEOUT",
    "ENV_SHENJI_URL",
    "ENV_SHENJI_TIMEOUT",
    "shenji_backend_config",
    "ENV_SHENJI_DB_PATH",
    "ENV_DATA_API_URL",
    "ENV_DATA_CACHE_DIR",
    "ENV_DATA_TIMEOUT",
    "ENV_DATA_SOURCE",
    "ENV_DUKASCOPY_USER",
    "ENV_DUKASCOPY_PASS",
    "DEFAULT_DATA_TIMEOUT",
    "DataAcquisitionConfig",
    "data_acquisition_config",
    "_default_shenji_db_path",
    "ENV_KLINE_DIR",
    "DEFAULT_KLINE_DIR",
    "kline_data_dir",
    "ENV_BUNDLED",
    "is_bundled",
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


# ── 妙算 后端（03 实时页「只读接入」）────────────────────────────────────
#
# 以下函数委托给统一设置系统（settings.py），同时保持向后兼容的函数签名。
# 环境变量仍然只在本模块（和 cli.py）被读取——settings.py 的 _load_env_config()
# 负责解析 MIAOSUAN_* 环境变量，本模块只是适配层。

#: 03 实时页默认后端地址（妙算 dashboard）——从统一设置系统获取默认值。
DEFAULT_SHENJI_URL: str = "http://127.0.0.1:1783"

#: 03 实时页默认请求超时（秒）。**必须短**：后端卡住不能把页面拖死。
DEFAULT_SHENJI_TIMEOUT: float = 3.0

#: 环境变量名：实时页后端地址 / 超时。
ENV_SHENJI_URL: str = "MIAOSUAN_SHENJI_URL"
ENV_SHENJI_TIMEOUT: str = "MIAOSUAN_SHENJI_TIMEOUT"


def shenji_backend_config(env: Mapping[str, str] | None = None) -> tuple[str, float]:
    """读取 03 实时页后端配置。

    优先从统一设置系统获取值（支持 YAML + 环境变量 + 热重载）；
    当传入 ``env`` 参数时（测试场景），回退到旧逻辑直接从 env 映射读取。

    Args:
        env: 覆盖用的环境映射（默认读 ``os.environ``，便于测试注入）。

    Returns:
        ``(base_url, timeout)``。
    """
    if env is not None:
        # 测试场景：直接从传入的 env 读取，保持与旧实现完全一致
        base_url = env.get(ENV_SHENJI_URL, "") or DEFAULT_SHENJI_URL
        raw_timeout = env.get(ENV_SHENJI_TIMEOUT, "")
        try:
            timeout = float(raw_timeout) if raw_timeout else DEFAULT_SHENJI_TIMEOUT
        except ValueError:
            timeout = DEFAULT_SHENJI_TIMEOUT
        if timeout <= 0.0:
            timeout = DEFAULT_SHENJI_TIMEOUT
        return base_url, timeout

    # 生产场景：从统一设置系统获取
    cfg = get_config()
    host = cfg.shenji.host
    port = cfg.shenji.port
    timeout = cfg.shenji.timeout
    base_url = f"http://{host}:{port}"
    # 兼容显式设置的 MIAOSUAN_SHENJI_URL（完整 URL 覆盖）
    explicit_url = os.environ.get(ENV_SHENJI_URL, "").strip()
    if explicit_url:
        base_url = explicit_url
    # 兼容显式设置的 MIAOSUAN_SHENJI_TIMEOUT（覆盖统一设置系统的缓存值）
    raw_timeout = os.environ.get(ENV_SHENJI_TIMEOUT, "").strip()
    if raw_timeout:
        try:
            timeout = float(raw_timeout)
        except ValueError:
            timeout = DEFAULT_SHENJI_TIMEOUT
        if timeout <= 0.0:
            timeout = DEFAULT_SHENJI_TIMEOUT
    return base_url, timeout


# ── 数据获取（架构 §3.2）────────────────────────────────────────────────────
#
# 与 shenji_backend_config 同属「唯一 env 边界」：环境变量只在本模块读取，
# data/acquisition 通过调用 data_acquisition_config() 注入配置，绝不自行读 env
# （可执行断言见 tests/test_dependency_direction_adapters.py:
#  test_os_environ_only_in_config_and_cli）。

#: 数据获取相关环境变量名。
ENV_SHENJI_DB_PATH: str = "MIAOSUAN_SHENJI_DB_PATH"
ENV_DATA_API_URL: str = "MIAOSUAN_DATA_API_URL"
ENV_DATA_CACHE_DIR: str = "MIAOSUAN_DATA_CACHE_DIR"
ENV_DATA_TIMEOUT: str = "MIAOSUAN_DATA_TIMEOUT"
ENV_DATA_SOURCE: str = "MIAOSUAN_DATA_SOURCE"
ENV_DUKASCOPY_USER: str = "MIAOSUAN_DUKASCOPY_USER"
ENV_DUKASCOPY_PASS: str = "MIAOSUAN_DUKASCOPY_PASS"
ENV_MT4_BRIDGE_HOST: str = "MIAOSUAN_MT4_BRIDGE_HOST"
ENV_MT4_BRIDGE_PORT: str = "MIAOSUAN_MT4_BRIDGE_PORT"
ENV_MT4_TIME_BASE: str = "MIAOSUAN_MT4_TIME_BASE"

#: 默认网络超时（秒）。
DEFAULT_DATA_TIMEOUT: int = 30

#: MT4 Bridge（FreeMT4Bridge EA）默认监听地址与端口。
DEFAULT_MT4_BRIDGE_HOST: str = "127.0.0.1"
DEFAULT_MT4_BRIDGE_PORT: int = 23232

#: MT4 输出时间戳口径默认值（``"utc"`` 符合 panel 契约）。
DEFAULT_MT4_TIME_BASE: str = "utc"

#: 环境变量名：本地行情数据目录（Web UI 扫描 parquet/csv 的根）。
ENV_KLINE_DIR: str = "MIAOSUAN_KLINE_DIR"
#: 默认行情数据目录（用户本机放置 TradingView 拉取结果的固定路径）。
DEFAULT_KLINE_DIR: str = r"D:\K线数据"


def _int_or_default(raw: str, default: int) -> int:
    """把环境变量串解析为整数；空 / 非法 / 非正数一律回落 ``default``。"""
    raw = (raw or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


@dataclass(frozen=True)
class DataAcquisitionConfig:
    """数据获取模块配置（经依赖注入向下传递）。"""

    shenji_db_path: str = ""
    data_api_url: str = ""
    cache_dir: str = ""
    timeout: int = DEFAULT_DATA_TIMEOUT
    data_source: str = ""
    dukascopy_user: str = ""
    dukascopy_password: str = ""
    mt4_bridge_host: str = DEFAULT_MT4_BRIDGE_HOST
    mt4_bridge_port: int = DEFAULT_MT4_BRIDGE_PORT
    mt4_time_base: str = DEFAULT_MT4_TIME_BASE


def kline_data_dir(env: Mapping[str, str] | None = None) -> str:
    """返回本地行情数据目录（供 Web UI / 数据获取扫描 parquet、csv）。

    优先从统一设置系统获取值（支持 YAML + 环境变量 + 热重载）；
    当传入 ``env`` 参数时（测试场景），回退到旧逻辑。

    Args:
        env: 覆盖用的环境映射（默认读 ``os.environ``，便于测试注入）。

    Returns:
        K 线数据目录路径。
    """
    if env is not None:
        return (env.get(ENV_KLINE_DIR, "") or "").strip() or DEFAULT_KLINE_DIR

    return get_config().paths.kline


#: launcher（PyInstaller bundle）标记：置 "1" 时 ``sys.executable`` 即 launcher 本身。
ENV_BUNDLED: str = "MIAOSUAN_BUNDLED"


def is_bundled(env: Mapping[str, str] | None = None) -> bool:
    """是否运行在 PyInstaller 打包的 launcher exe 内。

    与 :func:`kline_data_dir` 同属「唯一 env 边界」：``webui`` 只调用本函数，
    **绝不**自行读 env（守护测试 ``tests/test_dependency_direction_adapters.py``）。

    Args:
        env: 覆盖用的环境映射（默认读 ``os.environ``，便于测试注入）。

    Returns:
        ``MIAOSUAN_BUNDLED == "1"`` 时为 ``True``——此时 ``_cli_process``
        需以 ``[sys.executable, *args]`` 形式 spawn（由 launcher 路由 CLI），
        而非 ``[sys.executable, "-m", "miaosuan.cli", *args]``。
    """
    source = os.environ if env is None else env
    return (source.get(ENV_BUNDLED, "") or "").strip() == "1"


def _default_shenji_db_path() -> str:
    """返回妙算本地库默认候选路径中第一个存在的文件，否则空串。

    候选顺序：
    1. 统一设置系统的 paths.shenji_db（settings.yaml 或环境变量 MIAOSUAN_SHENJI_DB_PATH）；
    2. 用户本机妙算库固定路径（项目 owner 实测路径，单机「开箱即用」）；
    3. 若设置 ``MIAOSUAN_SHENJI_ROOT``，则拼接 ``<root>/data/market_data.db``。
    """
    # 先检查统一设置系统
    cfg = get_config()
    if cfg.paths.shenji_db:
        if os.path.isfile(cfg.paths.shenji_db):
            return cfg.paths.shenji_db
        # 用户显式配置了路径但文件不存在，直接返回（让 fetcher 报错）
        return cfg.paths.shenji_db

    candidates: list[str] = [
        r"D:\backup\BaoBao\PythonProgram\xauusd\data\market_data.db",
    ]
    root = os.environ.get("MIAOSUAN_SHENJI_ROOT", "")
    if root:
        candidates.append(os.path.join(root, "data", "market_data.db"))
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return ""


def data_acquisition_config(env: Mapping[str, str] | None = None) -> DataAcquisitionConfig:
    """读取数据获取相关配置。

    优先从统一设置系统获取值（支持 YAML + 环境变量 + 热重载）；
    当传入 ``env`` 参数时（测试场景），回退到旧逻辑直接从 env 映射读取。

    Args:
        env: 覆盖用的环境映射（默认读 ``os.environ``，便于测试注入）。

    Returns:
        :class:`DataAcquisitionConfig`。
    """
    if env is not None:
        # 测试场景：直接从传入的 env 读取
        raw_timeout = env.get(ENV_DATA_TIMEOUT, "")
        try:
            timeout = int(raw_timeout) if raw_timeout else DEFAULT_DATA_TIMEOUT
        except ValueError:
            timeout = DEFAULT_DATA_TIMEOUT
        if timeout <= 0:
            timeout = DEFAULT_DATA_TIMEOUT
        explicit_db = env.get(ENV_SHENJI_DB_PATH, "")
        shenji_db_path = explicit_db if explicit_db else _default_shenji_db_path()
        return DataAcquisitionConfig(
            shenji_db_path=shenji_db_path,
            data_api_url=env.get(ENV_DATA_API_URL, ""),
            cache_dir=env.get(ENV_DATA_CACHE_DIR, ""),
            timeout=timeout,
            data_source=env.get(ENV_DATA_SOURCE, ""),
            dukascopy_user=env.get(ENV_DUKASCOPY_USER, ""),
            dukascopy_password=env.get(ENV_DUKASCOPY_PASS, ""),
            mt4_bridge_host=(env.get(ENV_MT4_BRIDGE_HOST, "") or "").strip()
            or DEFAULT_MT4_BRIDGE_HOST,
            mt4_bridge_port=_int_or_default(
                env.get(ENV_MT4_BRIDGE_PORT, ""), DEFAULT_MT4_BRIDGE_PORT
            ),
            mt4_time_base=(env.get(ENV_MT4_TIME_BASE, "") or "").strip() or DEFAULT_MT4_TIME_BASE,
        )

    # 生产场景：从统一设置系统获取基础配置，环境变量实时覆盖
    cfg = get_config()
    shenji_db = cfg.paths.shenji_db
    if not shenji_db:
        shenji_db = _default_shenji_db_path()

    # 环境变量实时读取（兼容 monkeypatch 测试 + 不需要 reload 单例）
    # 旧版环境变量名优先于 settings 系统（向后兼容）
    mt4_host = (os.environ.get(ENV_MT4_BRIDGE_HOST, "") or "").strip() or cfg.mt4.host
    mt4_port = _int_or_default(
        os.environ.get(ENV_MT4_BRIDGE_PORT, ""), cfg.mt4.port
    )
    mt4_time_base = (os.environ.get(ENV_MT4_TIME_BASE, "") or "").strip() or cfg.mt4.time_base
    data_api = os.environ.get(ENV_DATA_API_URL, "") or cfg.data.api_url
    cache_dir = os.environ.get(ENV_DATA_CACHE_DIR, "") or cfg.data.cache_dir or cfg.paths.data_cache
    data_source = os.environ.get(ENV_DATA_SOURCE, "") or cfg.data.source
    dukascopy_user = os.environ.get(ENV_DUKASCOPY_USER, "") or cfg.data.dukascopy_user
    dukascopy_password = os.environ.get(ENV_DUKASCOPY_PASS, "") or cfg.data.dukascopy_password

    # 超时：环境变量优先，否则用设置系统的值
    raw_timeout = os.environ.get(ENV_DATA_TIMEOUT, "")
    if raw_timeout:
        try:
            timeout = int(raw_timeout)
        except ValueError:
            timeout = cfg.data.timeout
    else:
        timeout = cfg.data.timeout
    if timeout <= 0:
        timeout = DEFAULT_DATA_TIMEOUT

    # 显式妙算库路径环境变量优先
    explicit_db = os.environ.get(ENV_SHENJI_DB_PATH, "")
    if explicit_db:
        shenji_db = explicit_db

    return DataAcquisitionConfig(
        shenji_db_path=shenji_db,
        data_api_url=data_api,
        cache_dir=cache_dir,
        timeout=timeout,
        data_source=data_source,
        dukascopy_user=dukascopy_user,
        dukascopy_password=dukascopy_password,
        mt4_bridge_host=mt4_host,
        mt4_bridge_port=mt4_port,
        mt4_time_base=mt4_time_base,
    )


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
