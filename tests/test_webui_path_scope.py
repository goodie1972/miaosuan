# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for full license text.

"""WebUI 文件作用域回归测试（「每个按键都点一遍」对抗检查沉淀下来的防线）。

覆盖的是同一类缺陷：**产物目录分家后，端点仍把文件名解析/扫描写死在单一目录**。
它们不会让服务崩，只会以「按钮点了没反应 / 报文件不存在 / 列表永远为空」出现，
所以只能靠端点级测试兜住：

* ``POST /api/verify``、``GET /api/file/{name}``：入参是 ``/api/strategies``
  返回的**纯文件名**，而策略 .py 落在 ``strategies/``；旧实现按 ``artifacts/``
  解析 → 校验键与下载键永远 404。
* ``GET /api/backtests``：回测结果落在 ``data/results/``；旧实现只扫
  ``artifacts/`` → 历史回测永远空列表。
* ``GET /api/strategies``：只该扫策略目录（混扫 artifacts 会把临时脚本当策略）。
* ``DELETE /api/spec/{file}``：旧实现 ``from .. import artifacts`` 引用不存在的
  包属性 → ImportError → 500，删除键失灵。
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from fastapi import HTTPException

from miaosuan.webui import server

#: 带 STRATEGY_MAGIC 的最小「导出策略」源码：该常量由导出模板强制写入，
#: 是 /api/strategies 判定「这是导出产物」的唯一判据。
_STRATEGY_SRC = 'STRATEGY_MAGIC = "661901"\n'


def _call(
    app: Any,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
) -> tuple[int, bytes]:
    """用内存 ASGI 调用 app，返回 ``(状态码, 原始响应体)``（不假设是 JSON）。"""
    body = b"" if payload is None else json.dumps(payload).encode("utf-8")
    path_only, _, query = path.partition("?")
    scope: dict[str, Any] = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path_only,
        "raw_path": path_only.encode("utf-8"),
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
    return int(status), raw


@pytest.fixture
def dirs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Path]:
    """三个产物目录各自独立落在 tmp 下（不碰仓库真实产物）。"""
    artifacts, strategies, results = (tmp_path / n for n in ("artifacts", "strategies", "results"))
    for d in (artifacts, strategies, results):
        d.mkdir()
    monkeypatch.setattr(server, "_ARTIFACTS", artifacts)
    monkeypatch.setattr(server, "_STRATEGIES", strategies)
    monkeypatch.setattr(server, "_RESULTS", results)
    # sys.modules 里放 None 会让 import 抛 ImportError → 删除端点走直接 unlink 分支，
    # 不会把 tmp 文件丢进真实回收站。
    monkeypatch.setitem(sys.modules, "send2trash", None)
    return {"artifacts": artifacts, "strategies": strategies, "results": results}


@pytest.fixture
def app(dirs: dict[str, Path]) -> Any:
    """在隔离目录之上重新构建 app（端点闭包在创建时读目录常量）。"""
    return server.create_app()


# ── _safe_py：导出策略 .py 的目录解析 ───────────────────────────────────────


def test_safe_py_prefers_strategies_dir(dirs: dict[str, Path]) -> None:
    (dirs["artifacts"] / "ghost.py").write_text(_STRATEGY_SRC, encoding="utf-8")
    target = dirs["strategies"] / "ghost.py"
    target.write_text(_STRATEGY_SRC, encoding="utf-8")
    assert server._safe_py("ghost.py") == target.resolve()


def test_safe_py_falls_back_to_legacy_artifacts(dirs: dict[str, Path]) -> None:
    legacy = dirs["artifacts"] / "old.py"
    legacy.write_text(_STRATEGY_SRC, encoding="utf-8")
    assert server._safe_py("old.py") == legacy.resolve()


def test_safe_py_rejects_traversal(dirs: dict[str, Path]) -> None:
    for bad in ("../evil.py", "sub\\evil.py", ".."):
        with pytest.raises(HTTPException) as exc:
            server._safe_py(bad)
        assert exc.value.status_code == 400


def test_safe_py_missing_is_404(dirs: dict[str, Path]) -> None:
    with pytest.raises(HTTPException) as exc:
        server._safe_py("nope.py")
    assert exc.value.status_code == 404


# ── 下载键 /api/file/{name} ─────────────────────────────────────────────────


def test_file_download_serves_strategy_py(app: Any, dirs: dict[str, Path]) -> None:
    name = "20261026_h1_xauusd.py"
    (dirs["strategies"] / name).write_text(_STRATEGY_SRC, encoding="utf-8")
    status, raw = _call(app, "GET", f"/api/file/{name}")
    assert status == 200
    assert b"STRATEGY_MAGIC" in raw


def test_file_download_serves_artifacts_json(app: Any, dirs: dict[str, Path]) -> None:
    (dirs["artifacts"] / "spec_x.json").write_text('{"payload": {}}', encoding="utf-8")
    status, raw = _call(app, "GET", "/api/file/spec_x.json")
    assert status == 200
    assert b"payload" in raw


def test_file_download_missing_is_404(app: Any, dirs: dict[str, Path]) -> None:
    status, _ = _call(app, "GET", "/api/file/nope.json")
    assert status == 404


# ── 校验键 /api/verify ──────────────────────────────────────────────────────


def test_verify_resolves_strategy_py(app: Any, dirs: dict[str, Path]) -> None:
    """回归：曾因按 artifacts/ 解析策略文件名而**永远 404**（校验键形同虚设）。

    真跑 CLI 子进程做 lint：最小策略源码没有 ``_TOKENS``，lint 会报问题、
    返回码非 0，但那属于校验结论；本测试只锁「路径解析这一关过了」= HTTP 200。
    """
    name = "20261026_h1_xauusd.py"
    (dirs["strategies"] / name).write_text(_STRATEGY_SRC, encoding="utf-8")
    status, raw = _call(app, "POST", "/api/verify", {"file": name})
    assert status == 200
    assert b"\xe6\x96\x87\xe4\xbb\xb6\xe4\xb8\x8d\xe5\xad\x98\xe5\x9c\xa8" not in raw  # 「文件不存在」


def test_verify_unknown_file_is_404(app: Any, dirs: dict[str, Path]) -> None:
    status, _ = _call(app, "POST", "/api/verify", {"file": "does_not_exist.py"})
    assert status == 404


# ── 策略列表 /api/strategies：只扫策略目录 ──────────────────────────────────


def test_strategies_lists_only_strategies_dir(app: Any, dirs: dict[str, Path]) -> None:
    (dirs["strategies"] / "real.py").write_text(_STRATEGY_SRC, encoding="utf-8")
    # artifacts/ 里同名同内容的「历史遗留」不应被列出（口径唯一）
    (dirs["artifacts"] / "legacy.py").write_text(_STRATEGY_SRC, encoding="utf-8")
    rows = json.loads(_call(app, "GET", "/api/strategies")[1])
    assert [r["file"] for r in rows] == ["real.py"]


def test_strategies_skips_non_exported_py(app: Any, dirs: dict[str, Path]) -> None:
    (dirs["strategies"] / "scratch.py").write_text("print(1)\n", encoding="utf-8")
    assert json.loads(_call(app, "GET", "/api/strategies")[1]) == []


# ── 回测历史 /api/backtests：结果目录优先 ───────────────────────────────────


def test_backtests_lists_results_dir(app: Any, dirs: dict[str, Path]) -> None:
    """回归：/api/backtest 把结果写进 data/results/，列表却只扫 artifacts/ → 永远空。"""
    payload = {
        "meta": {"symbol": "XAUUSD", "timeframe": "H1", "n_bars": 1234},
        "summary": {"sharpe": 1.5, "total_return": 0.42},
    }
    (dirs["results"] / "backtest_20261027_010203.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    rows = json.loads(_call(app, "GET", "/api/backtests")[1])
    assert [r["file"] for r in rows] == ["backtest_20261027_010203.json"]
    assert rows[0]["symbol"] == "XAUUSD"
    assert rows[0]["sharpe"] == pytest.approx(1.5)


def test_backtests_dedupes_same_name(app: Any, dirs: dict[str, Path]) -> None:
    payload = {"meta": {}, "summary": {}}
    text = json.dumps(payload)
    (dirs["results"] / "backtest_dup.json").write_text(text, encoding="utf-8")
    (dirs["artifacts"] / "backtest_dup.json").write_text(text, encoding="utf-8")
    rows = json.loads(_call(app, "GET", "/api/backtests")[1])
    assert [r["file"] for r in rows] == ["backtest_dup.json"]


# ── 删除键 /api/spec/{file} ─────────────────────────────────────────────────


def test_delete_spec_removes_artifact_json(app: Any, dirs: dict[str, Path]) -> None:
    """回归：旧实现 ``from .. import artifacts`` → ImportError → 500，删除键失灵。"""
    victim = dirs["artifacts"] / "spec_killme.json"
    victim.write_text('{"payload": {}}', encoding="utf-8")
    status, raw = _call(app, "DELETE", "/api/spec/spec_killme.json")
    assert status == 200, raw
    assert not victim.exists()


def test_delete_spec_missing_is_404_and_keeps_dir(app: Any, dirs: dict[str, Path]) -> None:
    """删除不存在的 spec → 404（而不是旧实现的 500）。"""
    status, _ = _call(app, "DELETE", "/api/spec/ghost.json")
    assert status == 404
