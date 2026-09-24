# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""03 实时页：妙算 后端**只读**客户端测试。

重点证明"不下单"是**可被测试证明**的，而不是写在注释里：
越界（非 GET / 非白名单路径 / 非法查询参数）时 ``_open``（唯一 socket 出口）
**根本不会被调用**。
"""

from __future__ import annotations

import json
import urllib.error
from typing import Any

import pytest

from miaosuan.adapters.shenji.realtime import (
    ALLOWED_PATHS,
    DEFAULT_BASE_URL,
    ShenjiReadOnlyClient,
    BackendUnreachable,
    ReadOnlyViolation,
    RealtimeOutcome,
    env_config,
    safe_call,
)


class _Resp:
    """最小 ``urlopen`` 响应桩。"""

    def __init__(self, payload: Any, status: int = 200) -> None:
        self._raw = json.dumps(payload).encode("utf-8")
        self.status = status

    def read(self) -> bytes:
        return self._raw

    def __enter__(self) -> _Resp:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


def _bind(
    monkeypatch: pytest.MonkeyPatch,
    *,
    payload: Any = None,
    exc: BaseException | None = None,
    status: int = 200,
    record: list[tuple[str, str]] | None = None,
) -> ShenjiReadOnlyClient:
    """把客户端的 ``_open`` 换成桩，并可选记录调用。"""
    client = ShenjiReadOnlyClient()

    def _open(request: Any) -> Any:
        if record is not None:
            record.append((request.get_method(), request.full_url))
        if exc is not None:
            raise exc
        return _Resp(payload, status)

    monkeypatch.setattr(client, "_open", _open)
    return client


# ── 白名单 / 方法约束（"绝不下单"的可证明性）────────────────────────────────

def test_whitelist_contains_only_read_paths() -> None:
    """白名单必须**恰好**是那 5 个只读端点，不含任何写接口。"""
    expected = frozenset(
        {
            "/api/engine/status",
            "/api/market/price",
            "/api/market/candles",
            "/api/signals",
            "/api/signals/latest",
        }
    )
    assert expected == ALLOWED_PATHS
    for forbidden in (
        "/api/engine/start",
        "/api/engine/stop",
        "/api/orders",
        "/api/trade",
        "/api/positions/close",
    ):
        assert forbidden not in ALLOWED_PATHS


def test_non_get_method_is_rejected_without_network(monkeypatch: pytest.MonkeyPatch) -> None:
    record: list[tuple[str, str]] = []
    client = _bind(monkeypatch, payload={}, record=record)
    for method in ("POST", "PUT", "DELETE", "PATCH", "post"):
        with pytest.raises(ReadOnlyViolation):
            client.request(method, "/api/engine/status")
    assert record == [], "越界方法绝不允许发出任何请求"


def test_non_whitelisted_path_is_rejected_without_network(monkeypatch: pytest.MonkeyPatch) -> None:
    record: list[tuple[str, str]] = []
    client = _bind(monkeypatch, payload={}, record=record)
    for path in ("/api/engine/start", "/api/orders", "/api/secret", "/", "/api/engine/status2"):
        with pytest.raises(ReadOnlyViolation):
            client.get(path)
    assert record == [], "越界路径绝不允许发出任何请求"


def test_unknown_query_key_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    record: list[tuple[str, str]] = []
    client = _bind(monkeypatch, payload={}, record=record)
    with pytest.raises(ReadOnlyViolation):
        client.get("/api/market/candles", {"symbol": "XAUUSD", "timeframe": "H1", "evil": "1"})
    # /api/signals/statu 不允许任何查询参数（后端返回 400，但我们更早拦）
    with pytest.raises(ReadOnlyViolation):
        client.get("/api/engine/status", {"foo": "bar"})
    assert record == []


def test_get_sends_get_and_parses_json(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {"bid": 4348.38, "ask": 4348.82, "spread": 0.44, "symbol": "XAUUSD"}
    record: list[tuple[str, str]] = []
    client = _bind(monkeypatch, payload=payload, record=record)
    got = client.market_price()
    assert got == payload
    assert record == [("GET", DEFAULT_BASE_URL + "/api/market/price")]


def test_base_url_is_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    record: list[tuple[str, str]] = []
    client = ShenjiReadOnlyClient(base_url="http://127.0.0.1:9999/")

    def _open(request: Any) -> Any:
        record.append((request.get_method(), request.full_url))
        return _Resp({"status": "running"})

    monkeypatch.setattr(client, "_open", _open)
    client.engine_status()
    assert record == [("GET", "http://127.0.0.1:9999/api/engine/status")]


def test_candles_query_is_url_encoded_and_clamped(monkeypatch: pytest.MonkeyPatch) -> None:
    record: list[tuple[str, str]] = []
    client = _bind(monkeypatch, payload=[], record=record)
    client.market_candles("XAUUSD", "H1", 3)
    client.market_candles("XAU USD", "H1", 999999)  # 超上限 → 钳到 2000；空格 → 编码
    assert record[0][1].endswith("/api/market/candles?symbol=XAUUSD&timeframe=H1&count=3")
    assert "count=2000" in record[1][1]
    assert "symbol=XAU+USD" in record[1][1]


def test_signals_latest_requires_strategy() -> None:
    client = ShenjiReadOnlyClient()
    with pytest.raises(ValueError, match="必须指定策略名"):
        client.signals_latest("")
    with pytest.raises(ValueError, match="必须指定策略名"):
        client.signals_latest("   ")


# ── 降级：超时 / 不可达 / HTTP 错误 / 非法 JSON ──────────────────────────────

def test_timeout_maps_to_timeout_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _bind(monkeypatch, exc=TimeoutError("timed out"))
    with pytest.raises(BackendUnreachable) as info:
        client.engine_status()
    assert info.value.reason == "timeout"


def test_url_error_maps_to_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _bind(monkeypatch, exc=urllib.error.URLError(ConnectionRefusedError("refused")))
    with pytest.raises(BackendUnreachable) as info:
        client.market_price()
    assert info.value.reason == "unreachable"


def test_url_error_wrapping_timeout_maps_to_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _bind(monkeypatch, exc=urllib.error.URLError(TimeoutError("slow")))
    with pytest.raises(BackendUnreachable) as info:
        client.market_price()
    assert info.value.reason == "timeout"


def test_http_error_carries_status(monkeypatch: pytest.MonkeyPatch) -> None:
    exc = urllib.error.HTTPError("http://x", 400, "bad", {}, None)  # type: ignore[arg-type]
    client = _bind(monkeypatch, exc=exc)
    with pytest.raises(BackendUnreachable) as info:
        client.signals_latest("abc")
    assert info.value.reason == "http_error"
    assert info.value.status == 400


def test_bad_json_maps_to_bad_json_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    client = ShenjiReadOnlyClient()

    class _Raw:
        status = 200

        def read(self) -> bytes:
            return b"<html>not json</html>"

        def __enter__(self) -> Any:
            return self

        def __exit__(self, *exc: object) -> bool:
            return False

    monkeypatch.setattr(client, "_open", lambda request: _Raw())
    with pytest.raises(BackendUnreachable) as info:
        client.engine_status()
    assert info.value.reason == "bad_json"


# ── safe_call：信封折叠 ─────────────────────────────────────────────────────

def test_safe_call_ok_envelope(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _bind(monkeypatch, payload={"status": "running"})
    outcome = safe_call(client.engine_status)
    assert isinstance(outcome, RealtimeOutcome)
    assert outcome.ok is True
    assert outcome.reason == "ok"
    assert outcome.data == {"status": "running"}


def test_safe_call_failure_envelope(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _bind(monkeypatch, exc=TimeoutError("slow"))
    outcome = safe_call(client.engine_status)
    assert outcome.ok is False
    assert outcome.reason == "timeout"
    assert outcome.error


def test_safe_call_reraises_readonly_violation() -> None:
    """白名单越界是**编程错误**，不能被 safe_call 吞掉。"""
    client = ShenjiReadOnlyClient()
    with pytest.raises(ReadOnlyViolation):
        safe_call(client.get, "/api/engine/start")


# ── 环境配置 ────────────────────────────────────────────────────────────────

def test_env_config_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MIAOSUAN_SHENJI_URL", raising=False)
    monkeypatch.delenv("MIAOSUAN_SHENJI_TIMEOUT", raising=False)
    assert env_config() == (DEFAULT_BASE_URL, 3.0)


def test_env_config_override_and_bad_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MIAOSUAN_SHENJI_URL", "http://127.0.0.1:1234")
    monkeypatch.setenv("MIAOSUAN_SHENJI_TIMEOUT", "2.5")
    assert env_config() == ("http://127.0.0.1:1234", 2.5)
    monkeypatch.setenv("MIAOSUAN_SHENJI_TIMEOUT", "not-a-number")
    assert env_config()[1] == 3.0
    monkeypatch.setenv("MIAOSUAN_SHENJI_TIMEOUT", "-1")
    assert env_config()[1] == 3.0
