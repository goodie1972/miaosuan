"""WebUI 仪表盘接口测试（``src/miaosuan/webui/server.py``）。

为什么不用 ``fastapi.testclient.TestClient``
--------------------------------------------
``TestClient`` 依赖 ``httpx``，而本仓库并未声明 / 安装它（web 栈只有
``fastapi`` + ``uvicorn``）。为避免引入新依赖，这里用 ``asyncio`` + 裸 ASGI
``(scope, receive, send)`` 三元组直接驱动 app：走的仍是**真实**的路由分发与
异常转换（``HTTPException`` → 状态码 + ``{"detail": ...}``），只把传输层换成内存。

覆盖（回归防线）

* 未知标的 + 无 market → **400**（缺陷根因：以前会把 CLI 子进程的 Python
  堆栈直接甩给前端用户看）
* 未知标的 + 显式 market → 放行（不挡非标品种）
* 已知标的 ``XAUUSD`` → 放行，且**不**注入 ``--market``（保持自动推断）
* ``/api/specs`` 能列出**不含** ``spec_id`` 键的 spec JSON（回归 511581e）
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from miaosuan.webui import server

# ── 零依赖 ASGI 客户端 ──────────────────────────────────────────────────────

def _call(
    app: Any,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
) -> tuple[int, Any]:
    """用内存 ASGI 调用 app，返回 ``(状态码, 解析后的 JSON)``。"""
    body = b"" if payload is None else json.dumps(payload).encode("utf-8")
    scope: dict[str, Any] = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("utf-8"),
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"host", b"testserver"),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode("ascii")),
        ],
        "client": ("testclient", 123),
        "server": ("testserver", 80),
    }
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    async def drive() -> None:
        await app(scope, receive, send)

    asyncio.run(drive())

    status = next(m["status"] for m in sent if m["type"] == "http.response.start")
    raw = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return int(status), (json.loads(raw.decode("utf-8")) if raw else None)


class _FakeProc:
    """``subprocess.Popen`` 替身：不真跑挖掘（很慢），但让 pump 线程正常收尾。"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs
        self.stdout: list[str] = []
        self.returncode = 0

    def wait(self, timeout: float | None = None) -> int:
        """立即"结束"，返回 0。"""
        return self.returncode


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> Any:
    """隔离产物目录 + 桩掉子进程，返回全新的 app。"""
    monkeypatch.setattr(server, "_ARTIFACTS", tmp_path)
    monkeypatch.setattr(server.subprocess, "Popen", _FakeProc)
    server._jobs._job = None
    created = server.create_app()
    yield created
    server._jobs._job = None


# ── 前置校验：未知标的必须拦成 400 ──────────────────────────────────────────

def test_unknown_symbol_without_market_returns_400(app: Any) -> None:
    """未知标的 + 无 market → 400，且文案给出可操作指引（而不是 Python 堆栈）。"""
    status, data = _call(app, "POST", "/api/mine", {
        "data": "data/TESTXAU_H1.parquet",
        "symbol": "TESTXAU",
        "timeframe": "H1",
    })
    assert status == 400
    detail = data["detail"]
    assert "市场画像" in detail
    assert "TESTXAU" in detail
    assert "XAUUSD" in detail


def test_empty_symbol_without_market_returns_400(app: Any) -> None:
    """标的与 market 都为空 → 同样拦成 400（pipeline 一样会报错）。"""
    status, data = _call(app, "POST", "/api/mine", {"data": "data/TESTXAU_H1.parquet"})
    assert status == 400
    assert "市场画像" in data["detail"]


def test_unknown_symbol_with_explicit_market_is_allowed(app: Any) -> None:
    """显式指定 market 时放行，并且必须透传给 CLI 子进程。"""
    status, data = _call(app, "POST", "/api/mine", {
        "data": "data/TESTXAU_H1.parquet",
        "symbol": "TESTXAU",
        "market": "FOREX_XAUUSD",
    })
    assert status == 200, data
    assert data["job_id"]
    assert "--market" in data["cmd"]
    assert "FOREX_XAUUSD" in data["cmd"]


def test_known_symbol_without_market_is_allowed(app: Any) -> None:
    """已知标的 XAUUSD 走自动推断，不应注入 --market。"""
    status, data = _call(app, "POST", "/api/mine", {
        "data": "data/TESTXAU_H1.parquet",
        "symbol": "XAUUSD",
    })
    assert status == 200, data
    assert "--market" not in data["cmd"]
    assert "--symbol" in data["cmd"]


# ── 单一数据源：不重复硬编码字符串 ──────────────────────────────────────────

def test_known_symbols_and_profiles_come_from_single_source() -> None:
    """已知标的 / 可选画像都从各自唯一来源派生，避免两处字符串漂移。"""
    from miaosuan.pipeline import _SYMBOL_PROFILE_FALLBACK

    assert server._SYMBOL_PROFILE_FALLBACK is _SYMBOL_PROFILE_FALLBACK
    assert "XAUUSD" in server._SYMBOL_HINT
    assert "BTCUSDT" in server._SYMBOL_HINT
    for name in server.EXPECTED_PROFILE_NAMES:
        assert name in server._PROFILE_HINT


# ── 回归：/api/specs 不能因为缺 spec_id 键就吞掉历史产物 ───────────────────

def test_specs_lists_spec_without_spec_id_key(app: Any, tmp_path: Any) -> None:
    """只含 ``payload`` + ``spec_version``（**无** ``spec_id``）的 spec 也要能列出。

    回归 511581e：spec 列表曾用错的 schema 过滤，导致历史产物全部不可见。
    """
    (tmp_path / "spec_no_id.json").write_text(
        json.dumps({
            "payload": {"kind": "factor", "tokens": [51, 121, 53]},
            "spec_version": "1.0",
        }),
        encoding="utf-8",
    )
    status, data = _call(app, "GET", "/api/specs")
    assert status == 200
    files = [row["file"] for row in data]
    assert "spec_no_id.json" in files
    row = next(r for r in data if r["file"] == "spec_no_id.json")
    assert row["spec_id"]          # 列表接口自己派生短 id，不依赖文件里的键
    assert row["n_tokens"] == 3
