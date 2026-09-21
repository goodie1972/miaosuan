# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""DukascopyFetcher「连通性探测 + 错误透传」的针对性测试（**离线、秒级**）。

覆盖本次三处改动的契约：

1. ``is_available()`` 做**带缓存**的 TCP 可达性探测：只解析 IPv4 并连第一个地址，
   探测成功/失败两种结果都被缓存，且 socket **必须关闭**（无资源泄漏）；
2. ``describe()`` 在主机不可达时返回含「主机不可达」的字符串；
3. ``_fetch_raw()`` 在 HTTP 抛 ``DataError`` 时**向上抛出**，
   不再静默 ``break`` 把「网络不通」伪装成「无数据」。

全程用 ``monkeypatch`` 替换 ``socket.getaddrinfo`` / ``socket.create_connection``
与 ``_http_get``，**不发起任何真实网络请求**，因此离线环境同样稳定可跑。
"""

from __future__ import annotations

import socket

import pytest

from miaosuan.data.fetchers.dukascopy import DUKASCOPY_HOST, DukascopyFetcher
from miaosuan.errors import DataError

#: ``getaddrinfo`` 返回的 5 元组形态：``(family, type, proto, canonname, sockaddr)``
#: 探测代码只用到 ``info[4]``（sockaddr）与列表的第一个元素。
_ADDR_INFO = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("203.0.113.7", 443))]


class _FakeSock:
    """假 socket：只需记录是否被关闭（用于验证无资源泄漏）。"""

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _patch_socket(
    monkeypatch: pytest.MonkeyPatch,
    *,
    reachable: bool,
    dns_calls: list[str],
    opened: list[_FakeSock],
) -> None:
    """把 ``socket`` 的探测入口换成可控桩。

    Args:
        reachable: ``True`` 模拟可连通；``False`` 模拟连接被拒 / 超时。
        dns_calls: 收集 ``getaddrinfo`` 被调用的 host，用于验证缓存。
        opened: 收集被创建的假 socket，用于验证关闭。
    """

    def fake_getaddrinfo(
        host: str,
        port: int,
        family: int = 0,
        socktype: int = 0,
        proto: int = 0,
        flags: int = 0,
    ) -> list[tuple[int, int, int, str, tuple[str, int]]]:
        dns_calls.append(host)
        return list(_ADDR_INFO)

    def fake_create_connection(
        address: tuple[str, int],
        timeout: float | None = None,
        source_address: tuple[str, int] | None = None,
    ) -> _FakeSock:
        if not reachable:
            raise OSError("simulated unreachable")
        sock = _FakeSock()
        opened.append(sock)
        return sock

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    monkeypatch.setattr(socket, "create_connection", fake_create_connection)


# ── 1) 可达路径 ────────────────────────────────────────────────────────────


def test_is_available_true_and_socket_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    dns_calls: list[str] = []
    opened: list[_FakeSock] = []
    _patch_socket(monkeypatch, reachable=True, dns_calls=dns_calls, opened=opened)

    fetcher = DukascopyFetcher()
    assert fetcher.is_available() is True
    # 只解析一次，且只探测第一个地址（避免 IPv6/IPv4 各等满超时）
    assert dns_calls == [DUKASCOPY_HOST]
    assert len(opened) == 1
    # socket 必须关闭：不允许资源泄漏
    assert opened[0].closed is True


def test_is_available_result_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """结果按实例缓存：第二次调用不再触发 getaddrinfo / create_connection。"""
    dns_calls: list[str] = []
    opened: list[_FakeSock] = []
    _patch_socket(monkeypatch, reachable=True, dns_calls=dns_calls, opened=opened)

    fetcher = DukascopyFetcher()
    assert fetcher.is_available() is True
    assert fetcher.is_available() is True
    assert fetcher.is_available() is True
    # 三次调用只探测了一次
    assert len(dns_calls) == 1
    assert len(opened) == 1
    assert fetcher._avail_cache is True


# ── 2) 不可达路径 ──────────────────────────────────────────────────────────


def test_is_available_false(monkeypatch: pytest.MonkeyPatch) -> None:
    dns_calls: list[str] = []
    opened: list[_FakeSock] = []
    _patch_socket(monkeypatch, reachable=False, dns_calls=dns_calls, opened=opened)

    fetcher = DukascopyFetcher()
    assert fetcher.is_available() is False
    assert dns_calls == [DUKASCOPY_HOST]
    # 连接失败时没有 socket 被创建，自然无需关闭
    assert opened == []


def test_is_available_false_is_also_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """不可达结果同样被缓存——否则每次都要空等 probe_timeout 秒。"""
    dns_calls: list[str] = []
    opened: list[_FakeSock] = []
    _patch_socket(monkeypatch, reachable=False, dns_calls=dns_calls, opened=opened)

    fetcher = DukascopyFetcher()
    assert fetcher.is_available() is False
    assert fetcher.is_available() is False
    assert len(dns_calls) == 1
    assert fetcher._avail_cache is False


def test_is_available_when_getaddrinfo_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """DNS 解析直接抛异常时也要优雅返回 False，不能冒泡。"""

    def boom(*args: object, **kwargs: object) -> list:
        raise OSError("simulated dns failure")

    monkeypatch.setattr(socket, "getaddrinfo", boom)
    fetcher = DukascopyFetcher()
    assert fetcher.is_available() is False


# ── 3) describe() ──────────────────────────────────────────────────────────


def test_describe_unreachable_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_socket(monkeypatch, reachable=False, dns_calls=[], opened=[])
    text = DukascopyFetcher().describe()
    assert "主机不可达" in text
    assert DUKASCOPY_HOST in text


def test_describe_reachable_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_socket(monkeypatch, reachable=True, dns_calls=[], opened=[])
    text = DukascopyFetcher().describe()
    assert "主机不可达" not in text
    assert "public history" in text


# ── 4) _fetch_raw() 错误透传 ───────────────────────────────────────────────


def test_fetch_raw_raises_on_dataerror(monkeypatch: pytest.MonkeyPatch) -> None:
    """HTTP 抛 DataError 时必须向上抛，不能静默返回空列表。"""
    fetcher = DukascopyFetcher()

    def boom(url: str) -> list:
        raise DataError("Dukascopy 请求失败: simulated network down")

    monkeypatch.setattr(fetcher, "_http_get", boom)

    with pytest.raises(DataError) as excinfo:
        fetcher._fetch_raw("XAUUSD", 3600)

    msg = str(excinfo.value)
    assert "Dukascopy 拉取中断" in msg
    assert "窗口" in msg


def test_fetch_raw_still_returns_empty_when_no_bars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """对照用例：窗口**真的没行情**时仍正常返回空列表，不抛异常。

    与上一条一起，保证「网络不通」与「无数据」可被区分。
    """
    fetcher = DukascopyFetcher(window_days=1)
    seen: list[str] = []

    def empty(url: str) -> list:
        seen.append(url)
        return []

    monkeypatch.setattr(fetcher, "_http_get", empty)

    rows = fetcher._fetch_raw("XAUUSD", 3600)
    assert rows == []
    assert seen  # 确实发起过请求，只是该窗口无数据
