"""AlgoForge 后端**只读**客户端（03 实时页）。

红线（用户明确要求，且其 MT4 正在跑实盘）
----------------------------------------
**只读，绝不下单。** 该约束在**客户端层**用白名单强制，而不是靠调用方自觉：

* 只允许 ``GET`` 方法；
* 只允许 :data:`ALLOWED_PATHS` 里的**精确**路径；
* 查询参数只允许 :data:`ALLOWED_QUERY_KEYS` 里登记的键。

任何越界都会在**发出 socket 请求之前**抛 :class:`ReadOnlyViolation`
（因此"不下单"可以被测试证明：越界时 ``urlopen`` 根本不会被调用）。

架构与依赖
----------
妙算 WebUI 后端 → 本客户端 → AlgoForge dashboard 后端（默认
``http://127.0.0.1:1783``，见其 ``dashboard/backend/main.py:332``）。
只用标准库 :mod:`urllib`（环境未安装 ``httpx``，不引入新依赖）。
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from ...config import (
    DEFAULT_ALGOFORGE_TIMEOUT,
    DEFAULT_ALGOFORGE_URL,
    algoforge_backend_config,
)

__all__ = [
    "ALLOWED_PATHS",
    "ALLOWED_QUERY_KEYS",
    "DEFAULT_BASE_URL",
    "DEFAULT_TIMEOUT",
    "AlgoforgeReadOnlyClient",
    "BackendUnreachable",
    "ReadOnlyViolation",
    "RealtimeOutcome",
    "env_config",
    "safe_call",
]

#: 默认后端地址（AlgoForge dashboard；可用环境变量覆盖）。
#: 见 :func:`miaosuan.config.algoforge_backend_config`（env 的唯一读取点）。
DEFAULT_BASE_URL: str = DEFAULT_ALGOFORGE_URL

#: 默认超时（秒）。**必须短**：后端卡住不能把我们的页面拖死。
DEFAULT_TIMEOUT: float = DEFAULT_ALGOFORGE_TIMEOUT

#: 只读白名单：**精确**路径（不含查询串）。
#: 明确排除写接口（``/api/engine/start``、``/api/orders`` 等一律不在表内）。
ALLOWED_PATHS: frozenset[str] = frozenset(
    {
        "/api/engine/status",
        "/api/market/price",
        "/api/market/candles",
        "/api/signals",
        "/api/signals/latest",
    }
)

#: 每个路径允许的查询参数名（其余一律拒绝）。
ALLOWED_QUERY_KEYS: dict[str, frozenset[str]] = {
    "/api/market/candles": frozenset({"symbol", "timeframe", "count"}),
    "/api/signals/latest": frozenset({"strategy"}),
}

#: candles 单次请求的最大根数（避免一把拉爆后端）。
_MAX_CANDLES: int = 2000


class ReadOnlyViolation(RuntimeError):
    """试图越过只读白名单（非 GET 方法 / 非白名单路径 / 非法查询参数）。"""


class BackendUnreachable(RuntimeError):
    """后端不可达 / 超时 / 非 2xx / 响应非法。

    Attributes:
        reason: 稳定原因码（``unreachable`` / ``timeout`` / ``http_error`` / ``bad_json``）。
        status: HTTP 状态码（仅 ``http_error`` 时有值）。
    """

    def __init__(self, message: str, *, reason: str = "unreachable", status: int | None = None):
        super().__init__(message)
        self.reason = reason
        self.status = status


@dataclass(frozen=True)
class RealtimeOutcome:
    """一次只读调用的结果信封（供 WebUI 直接序列化）。

    Attributes:
        ok: 是否成功拿到数据。
        data: 成功时的原始 JSON（失败为 ``None``）。
        reason: 结果码（``ok`` 或 :class:`BackendUnreachable` 的 reason）。
        error: 失败时的可读原因（成功为 ``""``）。
        status: HTTP 状态码（若有）。
    """

    ok: bool
    data: Any = None
    reason: str = "ok"
    error: str = ""
    status: int | None = None


def env_config() -> tuple[str, float]:
    """从环境变量读后端地址与超时。

    实际读取由 :func:`miaosuan.config.algoforge_backend_config` 完成（env 的
    唯一读取点，见架构铁律 ``test_os_environ_only_in_config_and_cli``）；本函数
    只是适配层的一个稳定入口。

    Returns:
        ``(base_url, timeout)``。超时非法时回落到 :data:`DEFAULT_TIMEOUT`。
    """
    return algoforge_backend_config()


def safe_call(func: Any, /, *args: Any, **kwargs: Any) -> RealtimeOutcome:
    """执行一次只读调用并把异常折叠成 :class:`RealtimeOutcome`。

    只读白名单越界（:class:`ReadOnlyViolation`）属于**编程错误**，不吞——直接抛，
    因为它意味着调用方写了非法路径，属于需要修的 bug，而非运行期降级。

    Args:
        func: 只读调用（如 ``client.market_price``）。
        *args: 位置参数。
        **kwargs: 关键字参数。

    Returns:
        成功的 :class:`RealtimeOutcome`，或带 ``reason`` 的失败信封。
    """
    try:
        data = func(*args, **kwargs)
    except ReadOnlyViolation:
        raise
    except BackendUnreachable as exc:
        return RealtimeOutcome(ok=False, data=None, reason=exc.reason, error=str(exc), status=exc.status)
    except (ValueError, KeyError, TypeError) as exc:
        return RealtimeOutcome(ok=False, data=None, reason="invalid_argument", error=str(exc))
    return RealtimeOutcome(ok=True, data=data, reason="ok", error="")


@dataclass
class AlgoforgeReadOnlyClient:
    """AlgoForge 后端只读客户端。

    Attributes:
        base_url: 后端根地址（末尾斜杠会被去掉）。
        timeout: 单次请求超时（秒）。
    """

    base_url: str = DEFAULT_BASE_URL
    timeout: float = DEFAULT_TIMEOUT
    _call_log: list[str] = field(default_factory=list, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        self.timeout = float(self.timeout)
        if self.timeout <= 0.0:
            self.timeout = DEFAULT_TIMEOUT

    # ── 底层：白名单校验 + 请求 ────────────────────────────────────────────
    def request(self, method: str, path: str, params: dict[str, Any] | None = None) -> Any:
        """校验白名单后发起**只读**请求。

        Args:
            method: HTTP 方法（**只接受 GET**）。
            path: 请求路径（必须在 :data:`ALLOWED_PATHS` 内）。
            params: 查询参数（键必须在 :data:`ALLOWED_QUERY_KEYS` 内）。

        Returns:
            解析后的 JSON。

        Raises:
            ReadOnlyViolation: 方法 / 路径 / 查询参数越界（**请求不会发出**）。
            BackendUnreachable: 网络、超时、非 2xx 或响应非 JSON。
        """
        verb = str(method).upper()
        if verb != "GET":
            raise ReadOnlyViolation(f"只读客户端仅允许 GET，收到 {method!r}")
        if path not in ALLOWED_PATHS:
            raise ReadOnlyViolation(f"路径不在只读白名单：{path!r}")
        query = {k: v for k, v in (params or {}).items() if v is not None}
        extra = set(query) - ALLOWED_QUERY_KEYS.get(path, frozenset())
        if extra:
            raise ReadOnlyViolation(f"查询参数不在白名单：{sorted(extra)}（{path}）")
        return self._fetch(self._build_url(path, query))

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """``GET`` 便捷入口（走 :meth:`request` 的同一套白名单）。"""
        return self.request("GET", path, params)

    def _build_url(self, path: str, params: dict[str, Any]) -> str:
        """拼出绝对 URL（查询串做百分号编码）。"""
        if not params:
            return self.base_url + path
        return self.base_url + path + "?" + urllib.parse.urlencode(params)

    def _open(self, request: urllib.request.Request) -> Any:
        """实际发请求（**唯一**的 socket 出口，便于测试注入）。"""
        return urllib.request.urlopen(request, timeout=self.timeout)  # noqa: S310 - 白名单已限定

    def _fetch(self, url: str) -> Any:
        """发 GET 并解析 JSON（把底层异常统一翻译成 :class:`BackendUnreachable`）。"""
        request = urllib.request.Request(url, method="GET")  # 显式 GET
        self._call_log.append(url)
        try:
            with self._open(request) as response:
                status = int(getattr(response, "status", 0) or 0)
                raw = response.read()
        except urllib.error.HTTPError as exc:
            raise BackendUnreachable(
                f"后端返回 HTTP {exc.code}", reason="http_error", status=int(exc.code)
            ) from exc
        except TimeoutError as exc:
            raise BackendUnreachable(f"请求超时（>{self.timeout}s）", reason="timeout") from exc
        except urllib.error.URLError as exc:
            inner = exc.reason
            reason = "timeout" if isinstance(inner, TimeoutError) else "unreachable"
            raise BackendUnreachable(f"后端不可达：{inner}", reason=reason) from exc
        except OSError as exc:
            raise BackendUnreachable(f"后端不可达：{exc}", reason="unreachable") from exc
        if status and status != 200:
            raise BackendUnreachable(f"后端返回 HTTP {status}", reason="http_error", status=status)
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BackendUnreachable(f"响应不是合法 JSON：{exc}", reason="bad_json") from exc

    # ── 高层：白名单内的具体读接口 ────────────────────────────────────────
    def engine_status(self) -> Any:
        """读引擎状态（``GET /api/engine/status``）。"""
        return self.get("/api/engine/status")

    def market_price(self) -> Any:
        """读当前报价（``GET /api/market/price``）。"""
        return self.get("/api/market/price")

    def market_candles(self, symbol: str, timeframe: str, count: int = 120) -> Any:
        """读 K 线（``GET /api/market/candles``）。

        Args:
            symbol: 品种（如 ``XAUUSD``）。
            timeframe: 周期（如 ``H1``）。
            count: 根数，钳制到 ``[1, 2000]``。

        Returns:
            平台返回的 K 线数组。
        """
        bounded = max(1, min(int(count), _MAX_CANDLES))
        return self.get(
            "/api/market/candles",
            {"symbol": symbol, "timeframe": timeframe, "count": bounded},
        )

    def signals(self) -> Any:
        """读信号列表（``GET /api/signals``）。"""
        return self.get("/api/signals")

    def signals_latest(self, strategy: str) -> Any:
        """读指定策略的最新信号（``GET /api/signals/latest``）。

        Args:
            strategy: 策略名（**必填**；后端不传会返回 400）。

        Returns:
            后端返回的 JSON。

        Raises:
            ValueError: ``strategy`` 为空。
        """
        name = str(strategy or "").strip()
        if not name:
            raise ValueError("必须指定策略名（strategy）")
        return self.get("/api/signals/latest", {"strategy": name})
