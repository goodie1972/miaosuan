# -*- coding: utf-8 -*-
"""
统一配置系统（架构 §9.2 扩展）。

设计目标：
- 单一配置来源：YAML 文件（默认 settings.yaml）+ 环境变量覆盖
- 支持热重载：文件监听器自动检测变更并通知订阅者
- 类型安全：使用 Pydantic 模型进行验证和序列化
- 默认值保留：所有现有硬编码值作为后备默认值
- 无副作用导入：import 时不读取任何文件或环境变量
- 显式边界：只有通过 get_settings() 或 SettingsManager 获取配置

配置层级（从低到高优先级）：
1. 硬编码默认值（保持向后兼容）
2. YAML 配置文件（settings.yaml）
3. 环境变量（MIAOSUAN_* 前缀）
4. 运行时覆盖（仅用于测试和特殊场景）

所有配置通过 SettingsManager 单例访问，支持：
- settings.get_config() 获取当前配置快照
- settings.on_change(callback) 注册变更回调
- settings.update_overrides(**kwargs) 临时覆盖（测试用）
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

if TYPE_CHECKING:
    from watchdog.events import FileSystemEventHandler

# 尝试导入 watchdog 用于文件监听（mypy 友好：别名隔离 Any 基类）
try:
    from watchdog.events import FileSystemEventHandler as _FileSystemEventHandler
    from watchdog.observers import Observer
    WATCHDOG_AVAILABLE = True
except ImportError:  # pragma: no cover
    WATCHDOG_AVAILABLE = False

    class _FileSystemEventHandler:  # type: ignore[no-redef]
        pass

    class Observer:  # type: ignore[no-redef]
        pass

from .errors import ConfigError


# 默认值（保持与现有代码兼容）
DEFAULT_SETTINGS = {
    # 服务器端口
    "webui": {
        "port": 8686,  # WebUI 后端端口
        "host": "127.0.0.1",
        "reload": False,  # 开发时热重载
    },
    "shenji": {
        "port": 1783,  # 神机仪表盘端口（供 WebUI 只读访问）
        "host": "127.0.0.1",
        "timeout": 3.0,  # 请求超时（秒）
    },
    # MT4 Bridge 配置
    "mt4": {
        "host": "127.0.0.1",
        "port": 23232,  # FreeMT4Bridge EA 默认端口
        "time_base": "utc",  # utc / broker / shanghai
        "timeout": 5.0,  # 连接超时
        "poll_wait": 1000,  # EA 响应轮询间隔（ms）
        "drain_wait": 100,  # 数据排空等待（ms）
    },
    # 数据目录
    "paths": {
        "data_cache": "data/cache",  # OHLCV 数据缓存（parquet）
        "artifacts": "artifacts",  # Spec/策略/回测/调优产物
        "tmp": "tmp",  # 临时文件目录
        "kline": r"D:\K线数据",  # 本地 K 线数据根目录（TradingView 导出）
        "shenji_db": "",  # 妙算本地库路径（留空则自动探测）
    },
    # 数据源配置
    "data": {
        "timeout": 30,  # 网络请求超时（秒）
        "source": "",  # 默认数据源（"shenji"、"tradingview"等，空=自动）
        "api_url": "",  # 通用 HTTP 数据源地址
        "cache_dir": "",  # 数据缓存目录（留空则用 paths.data_cache）
        "dukascopy_user": "",  # Dukascopy 实盘用户名
        "dukascopy_password": "",  # Dukascopy 实盘密码
    },
    # 日志
    "logging": {
        "level": "INFO",  # DEBUG/INFO/WARNING/ERROR/CRITICAL
        "json_lines": True,  # 生产环境使用 JSON 行
    },
    # 其他功能开关
    "features": {
        "enable_webui": True,
        "enable_mt4": True,
        "enable_shenji": True,
        "enable_spec_management": True,
        "enable_backtest": True,
        "enable_tune": True,
    },
}


def _deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """递归合并两个字典，update 中的值优先"""
    result = base.copy()
    for key, value in update.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def _load_yaml_config(path: Path) -> dict[str, Any]:
    """从 YAML 文件加载配置"""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ConfigError(
            "YAML 支持需要安装 pyyaml: pip install pyyaml",
            context={"path": str(path)},
        ) from exc

    if not path.is_file():
        return {}

    try:
        with path.open("rt", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ConfigError(
                f"YAML 配置文件必须是映射类型，得到 {type(data).__name__}",
                context={"path": str(path), "type": type(data).__name__},
            )
        return data
    except yaml.YAMLError as exc:  # pragma: no cover
        raise ConfigError(
            f"YAML 配置文件解析失败: {exc}",
            context={"path": str(path)},
        ) from exc
    except OSError as exc:
        raise ConfigError(
            f"无法读取配置文件: {exc}",
            context={"path": str(path)},
        ) from exc


#: 旧版环境变量名 → 新版配置路径的映射表（向后兼容）。
#: 旧版 config.py 使用 MIAOSUAN_MT4_BRIDGE_HOST 等命名，
#: 新版 settings 使用 MIAOSUAN_MT4_HOST 等。此处把旧名映射到新结构。
_ENV_ALIASES: dict[str, tuple[str, ...]] = {
    "MIAOSUAN_SHENJI_URL": ("shenji",),       # 完整 URL，在 _apply_aliases 中特殊处理
    "MIAOSUAN_SHENJI_TIMEOUT": ("shenji", "timeout"),
    "MIAOSUAN_SHENJI_DB_PATH": ("paths", "shenji_db"),
    "MIAOSUAN_MT4_BRIDGE_HOST": ("mt4", "host"),
    "MIAOSUAN_MT4_BRIDGE_PORT": ("mt4", "port"),
    "MIAOSUAN_MT4_TIME_BASE": ("mt4", "time_base"),
    "MIAOSUAN_DATA_API_URL": ("data", "api_url"),
    "MIAOSUAN_DATA_CACHE_DIR": ("data", "cache_dir"),
    "MIAOSUAN_DATA_TIMEOUT": ("data", "timeout"),
    "MIAOSUAN_DATA_SOURCE": ("data", "source"),
    "MIAOSUAN_DUKASCOPY_USER": ("data", "dukascopy_user"),
    "MIAOSUAN_DUKASCOPY_PASS": ("data", "dukascopy_password"),
    "MIAOSUAN_KLINE_DIR": ("paths", "kline"),
    "MIAOSUAN_BUNDLED": ("_bundled",),        # 特殊标记，不进 Settings 模型
}


def _set_nested(config: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    """按路径在嵌套字典中设置值。"""
    current = config
    for part in path[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[path[-1]] = value


def _parse_env_value(value: str) -> Any:
    """把环境变量字符串转换为合适的类型。"""
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    if value.lstrip("-").isdigit():
        return int(value)
    try:
        return float(value)
    except ValueError:
        return value


def _load_env_config() -> dict[str, Any]:
    """从环境变量加载配置（MIAOSUAN_* 前缀）。

    支持两种命名约定：
    1. 新版直射：MIAOSUAN_WEBUI_PORT → webui.port, MIAOSUAN_MT4_HOST → mt4.host
    2. 旧版别名：MIAOSUAN_MT4_BRIDGE_HOST → mt4.host（经 _ENV_ALIASES 映射）
    """
    config: dict[str, Any] = {}
    prefix = "MIAOSUAN_"
    handled = set()  # 已通过别名处理的 key，跳过通用解析

    # 第一遍：处理旧版别名
    for key, value in os.environ.items():
        if key in _ENV_ALIASES:
            path = _ENV_ALIASES[key]
            if len(path) == 1 and path[0] == "shenji":
                # MIAOSUAN_SHENJI_URL 是完整 URL，拆成 host + port
                url_val = value.strip()
                if "://" in url_val:
                    # http://127.0.0.1:1783 → host=127.0.0.1, port=1783
                    import urllib.parse
                    parsed = urllib.parse.urlparse(url_val)
                    _set_nested(config, ("shenji", "host"), parsed.hostname or "127.0.0.1")
                    if parsed.port:
                        _set_nested(config, ("shenji", "port"), parsed.port)
                else:
                    _set_nested(config, ("shenji", "host"), url_val)
            elif len(path) == 1 and path[0] == "_bundled":
                # MIAOSUAN_BUNDLED 是特殊标记，不进 Settings 模型
                pass
            else:
                _set_nested(config, path, _parse_env_value(value))
            handled.add(key)

    # 第二遍：通用解析（新版命名 MIAOSUAN_WEBUI_PORT → webui.port）
    for key, value in os.environ.items():
        if not key.startswith(prefix) or key in handled:
            continue

        key_suffix = key[len(prefix):].lower()
        parts = key_suffix.split("_")
        if len(parts) < 2:
            continue

        current = config
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            elif not isinstance(current[part], dict):
                break
            current = current[part]
        else:
            current[parts[-1]] = _parse_env_value(value)

    return config


class WebUISettings(BaseModel):
    port: int = Field(default=8686, ge=1, le=65535)
    host: str = Field(default="127.0.0.1")
    reload: bool = Field(default=False)


class ShenjiSettings(BaseModel):
    port: int = Field(default=1783, ge=1, le=65535)
    host: str = Field(default="127.0.0.1")
    timeout: float = Field(default=3.0, gt=0)


class MT4Settings(BaseModel):
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=23232, ge=1, le=65535)
    time_base: Literal["utc", "broker", "shanghai"] = Field(default="utc")
    timeout: float = Field(default=5.0, gt=0)
    poll_wait: int = Field(default=1000, ge=0)  # ms
    drain_wait: int = Field(default=100, ge=0)  # ms


class PathsSettings(BaseModel):
    data_cache: str = Field(default="data/cache")
    artifacts: str = Field(default="artifacts")
    tmp: str = Field(default="tmp")
    kline: str = Field(default=r"D:\K线数据")
    shenji_db: str = Field(default="")  # 空字符串表示自动探测

    @model_validator(mode="after")
    def _expand_paths(self) -> "PathsSettings":
        """展开路径中的环境变量和 ~"""
        for field_name in self.model_fields:
            value = getattr(self, field_name)
            if isinstance(value, str):
                # 展开环境变量
                expanded = os.path.expandvars(value)
                # 展开用户目录
                expanded = os.path.expanduser(expanded)
                setattr(self, field_name, expanded)
        return self


class DataSettings(BaseModel):
    timeout: int = Field(default=30, gt=0)
    source: str = Field(default="")
    api_url: str = Field(default="")
    cache_dir: str = Field(default="")  # 空则使用 paths.data_cache
    dukascopy_user: str = Field(default="")
    dukascopy_password: str = Field(default="")


class LoggingSettings(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO"
    )
    json_lines: bool = Field(default=True)


class FeatureFlags(BaseModel):
    enable_webui: bool = Field(default=True)
    enable_mt4: bool = Field(default=True)
    enable_shenji: bool = Field(default=True)
    enable_spec_management: bool = Field(default=True)
    enable_backtest: bool = Field(default=True)
    enable_tune: bool = Field(default=True)


class Settings(BaseModel):
    """顶级配置模型"""
    webui: WebUISettings = Field(default_factory=WebUISettings)
    shenji: ShenjiSettings = Field(default_factory=ShenjiSettings)
    mt4: MT4Settings = Field(default_factory=MT4Settings)
    paths: PathsSettings = Field(default_factory=PathsSettings)
    data: DataSettings = Field(default_factory=DataSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    features: FeatureFlags = Field(default_factory=FeatureFlags)

    @model_validator(mode="after")
    def _resolve_cache_dir(self) -> "Settings":
        """解析数据缓存目录：如果 data.cache_dir 为空，则使用 paths.data_cache"""
        if not self.data.cache_dir:
            object.__setattr__(self.data, "cache_dir", self.paths.data_cache)
        return self


class SettingsManager:
    """配置管理器单例"""

    _instance: Optional["SettingsManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "SettingsManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._initialized = True

        self._config: Settings = Settings()  # 当前配置
        self._config_path: Optional[Path] = None
        self._observers: list[Callable[[Settings], None]] = []
        self._observer_lock = threading.Lock()
        self._reload_task: Optional[asyncio.Task[Any]] = None
        self._stop_event = threading.Event()

        # 初始加载
        self.reload()

    def _get_config_path(self) -> Path:
        """获取配置文件路径"""
        if self._config_path is not None:
            return self._config_path

        # 默认查找顺序：当前目录/settings.yaml -> 项目根目录/settings.yaml
        cwd = Path.cwd()
        for path in [cwd / "settings.yaml", cwd.parent / "settings.yaml"]:
            if path.is_file():
                self._config_path = path
                break
        else:
            # 默认使用当前目录
            self._config_path = cwd / "settings.yaml"

        return self._config_path

    def reload(self) -> Settings:
        """重新加载配置（YAML + 环境变量）"""
        # 按优先级加载：默认值 <- YAML <- 环境变量
        config_dict = DEFAULT_SETTINGS.copy()

        # 1. 加载 YAML 配置
        try:
            yaml_data = _load_yaml_config(self._get_config_path())
            config_dict = _deep_update(config_dict, yaml_data)
        except ConfigError:
            # YAML 加载失败但文件存在时才报错；不存在则忽略
            if self._get_config_path().is_file():
                raise

        # 2. 加载环境变量配置（最高优先级）
        env_data = _load_env_config()
        config_dict = _deep_update(config_dict, env_data)

        # 3. 创建并验证配置对象
        try:
            new_config = Settings.model_validate(config_dict)
        except ValidationError as exc:
            raise ConfigError(f"配置验证失败: {exc}") from exc

        old_config = self._config
        self._config = new_config

        # 通知所有观察者
        with self._observer_lock:
            observers = self._observers.copy()
        for observer in observers:
            try:
                observer(new_config)
            except Exception as exc:  # pragma: no cover
                # 单个观察者失败不应影响其他观察者
                print(f"Settings observer error: {exc}")

        return new_config

    def get_config(self) -> Settings:
        """获取当前配置的不可变副本"""
        return self._config

    def on_change(self, callback: Callable[[Settings], None]) -> Callable[[], None]:
        """
        注册配置变更回调

        返回一个可调用的对象，调用时取消注册
        """
        with self._observer_lock:
            self._observers.append(callback)

        def unsubscribe() -> None:
            with self._observer_lock:
                if callback in self._observers:
                    self._observers.remove(callback)

        return unsubscribe

    def update_overrides(self, **kwargs: Any) -> Settings:
        """
        临时更新配置（主要用于测试）
        注意：这不会持久化到文件，且不会触发文件监听器
        """
        current_dict = self._config.model_dump()
        updated_dict = _deep_update(current_dict, kwargs)
        try:
            new_config = Settings.model_validate(updated_dict)
        except ValidationError as exc:
            raise ConfigError(f"配置覆盖验证失败: {exc}") from exc

        old_config = self._config
        self._config = new_config

        # 通知观察者
        with self._observer_lock:
            observers = self._observers.copy()
        for observer in observers:
            try:
                observer(new_config)
            except Exception as exc:  # pragma: no cover
                print(f"Settings observer error: {exc}")

        return new_config

    def start_file_watcher(self) -> None:
        """启动文件系统监听器以实现热重载"""
        if not WATCHDOG_AVAILABLE:  # pragma: no cover
            print("Watchdog 未安装，无法启动文件监听器。"
                  " pip install watchdog 以启用热重载功能")
            return

        if self._reload_task is not None and not self._reload_task.done():
            return  # 已经在运行

        config_path = self._get_config_path()
        if not config_path.parent.is_dir():
            config_path.parent.mkdir(parents=True, exist_ok=True)

        class SettingsChangeHandler(_FileSystemEventHandler):  # type: ignore[misc]
            def __init__(self, manager: "SettingsManager") -> None:
                self.manager = manager
                self._last_reload = 0.0
                self._debounce_seconds = 1.0  # 防抖，避免频繁重载

            def on_modified(self, event: Any) -> None:
                if event.is_directory:
                    return
                if Path(event.src_path) != config_path.resolve():
                    return

                now = time.time()
                if now - self._last_reload < self._debounce_seconds:
                    return
                self._last_reload = now

                # 在后台线程中重载配置
                try:
                    self.manager.reload()
                    print(f"Settings reloaded due to change in {config_path}")
                except Exception as exc:  # pragma: no cover
                    print(f"Failed to reload settings: {exc}")

        self._observer = SettingsChangeHandler(self)
        self._watchdog_observer = Observer()
        self._watchdog_observer.schedule(
            self._observer, str(config_path.parent), recursive=False
        )
        self._watchdog_observer.start()

    def stop_file_watcher(self) -> None:
        """停止文件系统监听器"""
        if hasattr(self, "_watchdog_observer") and self._watchdog_observer is not None:
            self._watchdog_observer.stop()
            self._watchdog_observer.join()
            self._watchdog_observer = None
        if hasattr(self, "_observer"):
            self._observer = None  # type: ignore[assignment]

    def shutdown(self) -> None:
        """关闭管理器，清理资源"""
        self.stop_file_watcher()
        self._stop_event.set()


# 全局单例实例
_settings_manager: Optional[SettingsManager] = None
_manager_lock = threading.Lock()


def get_settings_manager() -> SettingsManager:
    """获取配置管理器单例"""
    global _settings_manager
    with _manager_lock:
        if _settings_manager is None:
            _settings_manager = SettingsManager()
        return _settings_manager


def get_config() -> Settings:
    """获取当前配置的便捷函数"""
    return get_settings_manager().get_config()


def on_config_change(callback: Callable[[Settings], None]) -> Callable[[], None]:
    """注册配置变更回调的便捷函数"""
    return get_settings_manager().on_change(callback)


def update_config_overrides(**kwargs: Any) -> Settings:
    """更新配置覆盖的便捷函数（主要用于测试）"""
    return get_settings_manager().update_overrides(**kwargs)


def reload_config() -> Settings:
    """重新加载配置的便捷函数"""
    return get_settings_manager().reload()


def start_settings_watcher() -> None:
    """启动设置文件监听器的便捷函数"""
    get_settings_manager().start_file_watcher()


def stop_settings_watcher() -> None:
    """停止设置文件监听器的便捷函数"""
    get_settings_manager().stop_file_watcher()


# 为了向后兼容，提供一些常用的直接访问函数
def get_webui_port() -> int:
    return get_config().webui.port


def get_mt4_port() -> int:
    return get_config().mt4.port


def get_data_cache_dir() -> str:
    cfg = get_config()
    return cfg.data.cache_dir or cfg.paths.data_cache


def get_artifacts_dir() -> str:
    return get_config().paths.artifacts


def get_tmp_dir() -> str:
    return get_config().paths.tmp


def get_kline_dir() -> str:
    return get_config().paths.kline


def get_shenji_db_path() -> str:
    return get_config().paths.shenji_db


# 导出公共接口
__all__ = [
    "Settings",
    "SettingsManager",
    "get_settings_manager",
    "get_config",
    "on_config_change",
    "update_config_overrides",
    "reload_config",
    "start_settings_watcher",
    "stop_settings_watcher",
    "get_webui_port",
    "get_mt4_port",
    "get_data_cache_dir",
    "get_artifacts_dir",
    "get_tmp_dir",
    "get_kline_dir",
    "get_shenji_db_path",
    "ConfigError",
]