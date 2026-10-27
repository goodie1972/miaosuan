# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""``POST /api/export`` 的 ``gate_deadband`` → CLI 实参映射（GATE 死区下沉的 UI 侧）。

前端目前只发 ``{spec, name}``（不带死区），所以缺省语义必须是**跟随 spec**：
不能把 ``--gate-deadband`` 拼进去，否则会把 ``0.0`` 硬塞给本应读
``semantics.gate_deadband`` 的导出。反过来，显式给值必须透传，且
**显式 0.0 不能被当成 falsy 丢掉**（0 = 明确要求关闭保护，与 None 不同义）。

沿用 ``tests/test_webui.py`` 的零依赖内存 ASGI 驱动（仓库未装 httpx）。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from miaosuan.webui import server


def _call(app: Any, path: str, payload: dict[str, Any]) -> tuple[int, Any]:
    """内存 ASGI 调用，返回 ``(状态码, 解析后的 JSON)``。"""
    body = json.dumps(payload).encode("utf-8")
    scope: dict[str, Any] = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
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


class _CapturedRun:
    """``subprocess.run`` 替身：只记录 argv，绝不真跑 CLI。"""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, argv: Any, *args: Any, **kwargs: Any) -> Any:
        self.calls.append([str(a) for a in (argv if isinstance(argv, (list, tuple)) else [argv])])
        return type("Proc", (), {"returncode": 0, "stdout": b"ok", "stderr": b""})()

    def flag(self) -> str | None:
        """最近一次调用里 ``--gate-deadband`` 的取值（不存在则 None）。"""
        argv = self.calls[-1]
        return argv[argv.index("--gate-deadband") + 1] if "--gate-deadband" in argv else None


@pytest.fixture
def export_app(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[Any, _CapturedRun]:
    """隔离产物目录 + 捕获子进程实参，返回 ``(app, 捕获器)``。"""
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "spec_probe.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(server, "_ARTIFACTS", artifacts)
    monkeypatch.setattr(server, "_STRATEGIES", tmp_path / "strategies")
    captured = _CapturedRun()
    monkeypatch.setattr(server.subprocess, "run", captured)
    yield server.create_app(), captured


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        # 前端现状：只发 spec → 不带死区实参 → CLI 跟随 semantics.gate_deadband
        ({"spec": "spec_probe.json"}, None),
        ({"spec": "spec_probe.json", "name": ""}, None),
        # 显式数值：必须透传
        ({"spec": "spec_probe.json", "gate_deadband": 0.03}, "0.03"),
        # 显式 0（明确要求关闭）≠ 缺省：不能因 falsy 被丢掉
        ({"spec": "spec_probe.json", "gate_deadband": 0.0}, "0.0"),
        ({"spec": "spec_probe.json", "gate_deadband": None}, None),
    ],
)
def test_export_maps_gate_deadband_to_cli(
    export_app: tuple[Any, _CapturedRun], payload: dict[str, Any], expected: str | None
) -> None:
    app, captured = export_app
    status, data = _call(app, "/api/export", payload)
    assert status == 200, data
    assert captured.calls, "导出端点未拼出 CLI 实参"
    assert captured.flag() == expected, f"--gate-deadband 映射错误：{captured.calls[-1]}"
    assert "export" in captured.calls[-1]


def test_export_rejects_unknown_field_type(export_app: tuple[Any, _CapturedRun]) -> None:
    """死区必须是数值：字符串不能悄悄变成 0.0 或被忽略（422 由 pydantic 负责）。"""
    app, captured = export_app
    status, _data = _call(app, "/api/export", {"spec": "spec_probe.json", "gate_deadband": "wide"})
    assert status == 422
    assert captured.calls == []
