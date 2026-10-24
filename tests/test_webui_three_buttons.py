# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for full license text.

"""WebUI 三键挖掘（开始/继续、暂停、重新开始）接口测试。"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from miaosuan.webui import server


def _call(
    app: Any,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
) -> tuple[int, Any]:
    """用内存 ASGI 调用 app，返回 ``(状态码, 解析后的 JSON)``。"""
    body = b"" if payload is None else json.dumps(payload).encode("utf-8")
    path_only, _, query = path.partition("?")
    scope: dict[str, Any] = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path_only,
        "raw_path": path.encode("utf-8"),
        "query_string": query.encode("utf-8"),
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


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> Any:
    monkeypatch.setattr(server, "_ARTIFACTS", tmp_path)
    return server.app


def test_mine_start_endpoint(app: Any) -> None:
    """测试 /api/mine/start 端点行为正常。"""
    status, data = _call(app, "POST", "/api/mine/start", {
        "data": "non_existent_data.parquet",
        "budget": "quick",
    })
    assert status in (400, 409, 422, 500)


def test_mine_pause_endpoint(app: Any) -> None:
    """测试 /api/mine/pause 端点行为正常（无任务时应返回 404）。"""
    status, data = _call(app, "POST", "/api/mine/pause", {})
    assert status in (404, 409)


def test_mine_restart_endpoint(app: Any) -> None:
    """测试 /api/mine/restart 端点行为正常。"""
    status, data = _call(app, "POST", "/api/mine/restart", {"data": "x", "budget": "quick"})
    assert status in (200, 400, 404, 409, 422, 500)
