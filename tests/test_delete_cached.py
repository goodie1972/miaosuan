# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""``DELETE /api/acquisition/cached`` 删除缓存文件的安全契约测试。

守护 4 条语义：

1. **路径白名单**：只允许 ``_DATA_DIRS`` 内的文件，路径穿越（``../``）拒绝；
2. **幂等**：文件不存在也返回 200（``reason=not_found``），不报错；
3. **删除成功**：文件从磁盘消失，响应含 ``trashed`` 标志；
4. **非数据目录**的文件（如 artifacts/ 下）拒绝 400。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from miaosuan.webui.server import create_app


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """构造一个只扫 tmp_path 的 TestClient，不碰真实数据目录。

    _DATA_DIRS 在模块加载时固定（读 kline_data_dir()），monkeypatch env
    无法回溯修改它。改为直接 patch server 模块的 _DATA_DIRS。
    """
    fake_dir = tmp_path / "fake_kline"
    fake_dir.mkdir()
    import miaosuan.webui.server as srv

    monkeypatch.setattr(srv, "_DATA_DIRS", (fake_dir,), raising=True)
    app = srv.create_app()
    return TestClient(app)


def _make_file(directory: Path, name: str = "XAUUSD_D1.parquet") -> Path:
    """在目录里造一个假的 parquet 文件（内容无所谓，删除端点不读内容）。"""
    p = directory / name
    p.write_bytes(b"PAR1fake")
    return p


# ── 正常删除 ──────────────────────────────────────────────────────────────


def test_delete_success(client: TestClient, tmp_path: Path):
    """白名单内文件删除成功，文件从磁盘消失。"""
    fake_dir = tmp_path / "fake_kline"
    f = _make_file(fake_dir)
    assert f.is_file()

    resp = client.delete("/api/acquisition/cached", params={"path": str(f)})
    # 调试输出
    print(f"DEBUG: path={f}", flush=True)
    print(f"DEBUG: resp.status={resp.status_code}", flush=True)
    print(f"DEBUG: resp.body={resp.text}", flush=True)
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["trashed"] in (True, False)  # send2trash 装了 = True，没装 = False
    assert not f.is_file(), "文件应该已被删除"


def test_delete_idempotent_not_found(client: TestClient, tmp_path: Path):
    """文件不存在时返回 200 + reason=not_found（幂等，不 404）。"""
    fake_dir = tmp_path / "fake_kline"
    nonexistent = fake_dir / "never_existed.parquet"
    assert not nonexistent.is_file()

    resp = client.delete("/api/acquisition/cached", params={"path": str(nonexistent)})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["reason"] == "not_found"


# ── 路径安全 ──────────────────────────────────────────────────────────────


def test_delete_path_traversal_rejected(client: TestClient):
    """路径穿越（../）必须被 _resolve_data_file 拦截 → 400。"""
    resp = client.delete(
        "/api/acquisition/cached",
        params={"path": "../../../../etc/passwd"},
    )
    assert resp.status_code == 400


def test_delete_outside_data_dirs_rejected(client: TestClient, tmp_path: Path):
    """非数据目录内的文件拒绝删除（artifacts/ 下的文件不算缓存数据）。"""
    outside = tmp_path / "not_in_data_dirs.parquet"
    outside.write_bytes(b"PAR1fake")

    resp = client.delete("/api/acquisition/cached", params={"path": str(outside)})
    assert resp.status_code == 400
    assert "不允许的数据目录" in resp.json()["detail"] or "不在" in resp.json()["detail"]


def test_delete_empty_path_rejected(client: TestClient):
    """空路径 → 400。"""
    resp = client.delete("/api/acquisition/cached", params={"path": ""})
    assert resp.status_code == 400
