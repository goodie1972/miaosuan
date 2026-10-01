# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""``POST /api/config`` 回归守护：**None 不得穿透进合并结果**。

事故复现（2026-10-01）：设置页点「保存配置」报 8 个 pydantic 错误
（webui.port / shenji.port / shenji.timeout / mt4.* / data.timeout 全是
``input_value=None``）。链路：

1. 前端 ``parseInt("")`` = NaN（输入框空——loadConfig 未返回 / 加载失败）；
2. ``JSON.stringify(NaN)`` → ``null``（JSON 不允许 NaN，序列化降级）；
3. ``ConfigUpdateRequest`` 的 section 是 ``dict[str, Any]``——
   ``model_dump(exclude_none=True)`` **只剔顶层**，dict 内部的 None 穿透；
4. ``_deep_update`` 把 None 合进 yaml 镜像 → ``Settings.model_validate``
   对 int/float 字段报错 → 保存 400，用户看到 8 连挂。

修复双保险：

* **前端** ``getNum()``：NaN → ``undefined``，JSON.stringify 直接省略键；
* **后端** ``_strip_none``：合并前递归剥掉 dict 内部的 None——
  None = 「未提供」，绝不覆盖 yaml 原值（防御任何客户端，不只本前端）。

本测试守护后端这一层（前端层无法在 pytest 里测）。同时固化一条语义：
**未提供的字段保留 yaml 原值，提供的字段正常更新**。
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from miaosuan import settings as settings_mod
from miaosuan.webui.server import create_app


@pytest.fixture()
def isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """把配置文件隔离到 tmp_path，测试后恢复单例状态。

    端点内部走 ``get_settings_manager()._get_config_path()``——直接改单例的
    ``_config_path`` 即可让读写都落到 tmp；同时快照 ``_config``，测试结束
    复原（reload() 会替换它，不复原会污染同进程后续测试）。
    """
    manager = settings_mod.get_settings_manager()
    original_path = manager._config_path
    original_config = manager._config

    fake_yaml = tmp_path / "settings.yaml"
    fake_yaml.write_text(
        "webui:\n  port: 8686\n  host: 127.0.0.1\n"
        "shenji:\n  timeout: 3.0\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(manager, "_config_path", fake_yaml)

    yield fake_yaml

    manager._config_path = original_path
    manager._config = original_config


@pytest.fixture()
def client(isolated_config: Path):
    with TestClient(create_app()) as c:
        yield c


def _read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


# ── 1) 核心回归：dict 内部的 null 不得覆盖 yaml 原值 ────────────────────────


def test_null_numeric_fields_do_not_overwrite_yaml(client, isolated_config):
    """8 连挂事故复现：数值字段全发 null → 必须 200 且 yaml 原值保留。"""
    resp = client.post(
        "/api/config",
        json={
            "webui": {"port": None, "host": "10.0.0.1"},
            "shenji": {"timeout": None},
        },
    )
    assert resp.status_code == 200, resp.text

    data = _read_yaml(isolated_config)
    assert data["webui"]["port"] == 8686, "null 不得把端口清掉"
    assert data["webui"]["host"] == "10.0.0.1", "正常字段必须更新"
    assert data["shenji"]["timeout"] == 3.0, "null 不得把超时清掉"


def test_all_null_like_frontend_nan_payload(client, isolated_config):
    """模拟前端 NaN 序列化后的完整 payload（出事时的形状）→ 200 不 400。"""
    resp = client.post(
        "/api/config",
        json={
            "webui": {"host": "127.0.0.1", "port": None, "reload": False},
            "shenji": {"host": "127.0.0.1", "port": None, "timeout": None},
            "mt4": {
                "host": "127.0.0.1",
                "port": None,
                "time_base": "utc",
                "timeout": None,
                "poll_wait": None,
                "drain_wait": None,
            },
            "paths": {"kline": "D:\\K线数据"},
            "data": {
                "timeout": None,
                "source": "",
                "api_url": "",
                "cache_dir": "",
                "dukascopy_user": "",
                "dukascopy_password": "",
            },
        },
    )
    assert resp.status_code == 200, resp.text
    # 剥离 None 后 merged 只含合法值——再回读一次应同样通过校验
    assert client.get("/api/config").status_code == 200


# ── 2) 正常路径不回归 ───────────────────────────────────────────────────────


def test_normal_save_updates_and_persists(client, isolated_config):
    """正常数值保存 → 200，yaml 更新，GET 回读一致。"""
    resp = client.post("/api/config", json={"webui": {"port": 9999}})
    assert resp.status_code == 200, resp.text
    assert _read_yaml(isolated_config)["webui"]["port"] == 9999
    assert client.get("/api/config").json()["webui"]["port"] == 9999


def test_invalid_value_still_rejected(client, isolated_config):
    """防御不能过度：非法值（port 越界）仍必须 400，坏配置不写盘。"""
    resp = client.post("/api/config", json={"webui": {"port": 99999}})
    assert resp.status_code == 400
    assert _read_yaml(isolated_config)["webui"]["port"] == 8686


def test_empty_section_dict_is_noop(client, isolated_config):
    """空 dict section（全部字段都是 null 被剥光后）→ 200，等价未提供。"""
    resp = client.post("/api/config", json={"mt4": {"port": None}})
    assert resp.status_code == 200, resp.text
    assert "mt4" not in _read_yaml(isolated_config)  # 原 yaml 没有 mt4 节
