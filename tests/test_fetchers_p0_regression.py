# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""两处已修 P0 的**回归守护**（防再次退化）——全程离线、不发起真实网络请求。

1. **OKX 取错品种**：``OkxFetcher.fetch_full`` 曾丢弃入参 ``symbol``（``_ = symbol``），
   恒用构造期默认 ``XAUUSD``，导致 CRYPTO_BTC 画像下取 ``BTCUSDT`` 却静默返回
   黄金行情——不报错但给错数据。守护方式：断言**实际打出的请求 URL** 里
   ``instId`` 等于入参品种（而不是断言返回的 DataFrame，那样抓不到这个 Bug）。

2. **acquisition 吞掉失败原因**：``NetworkSource.fetch_full/fetch_incremental``
   曾是四处 ``except Exception: pass/continue``，把所有失败压成一句「无可用来源」，
   用户分不清「没装包 / 网络不通 / 品种不支持」。守护方式：断言每个来源的失败
   原因**同时**出现在错误消息与 ``DataError.context["failures"]`` 里。
"""

from __future__ import annotations

import urllib.parse

import pandas as pd
import pytest

from miaosuan.data.acquisition import NetworkSource
from miaosuan.data.fetchers import BaseFetcher
from miaosuan.data.fetchers.okx import OkxFetcher
from miaosuan.errors import DataError

# ── 1) OKX：品种必须由 fetch_full 入参决定 ─────────────────────────────────

#: 一根已收盘的 OKX 蜡烛：[ts_ms, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
_OKX_ROW = ["1700000000000", "1.0", "2.0", "0.5", "1.5", "100", "", "", "1"]


class _RecordingHttpGetJson:
    """替换 ``BaseFetcher._http_get_json``：记录请求 URL 并返回固定 OKX 响应。"""

    def __init__(self) -> None:
        self.urls: list[str] = []

    def __call__(
        self,
        url: str,
        timeout: float = 20.0,  # noqa: ARG002
        *,
        headers: dict[str, str] | None = None,  # noqa: ARG002
    ) -> dict:
        self.urls.append(url)
        return {"code": "0", "msg": "", "data": [list(_OKX_ROW)]}


def _inst_id_of(url: str) -> str:
    """从请求 URL 里取出 ``instId`` 参数。"""
    query = urllib.parse.urlparse(url).query
    return urllib.parse.parse_qs(query).get("instId", [""])[0]


def test_okx_request_uses_symbol_argument(monkeypatch: pytest.MonkeyPatch) -> None:
    """核心断言：请求参数必须是入参品种（而非构造期默认值）。"""
    recorder = _RecordingHttpGetJson()
    monkeypatch.setattr(BaseFetcher, "_http_get_json", staticmethod(recorder))

    fetcher = OkxFetcher()  # 构造期默认 XAUUSD
    df = fetcher.fetch_full("BTCUSDT", "H1")

    assert recorder.urls, "应至少发起一次 HTTP 请求"
    assert _inst_id_of(recorder.urls[0]) == "BTCUSDT"
    assert not df.empty


@pytest.mark.parametrize("blank", ["", "   ", None])
def test_okx_falls_back_to_default_only_when_symbol_blank(
    monkeypatch: pytest.MonkeyPatch, blank: str | None
) -> None:
    """入参为空才回落构造期默认品种（XAUUSD）。"""
    recorder = _RecordingHttpGetJson()
    monkeypatch.setattr(BaseFetcher, "_http_get_json", staticmethod(recorder))

    fetcher = OkxFetcher()
    fetcher.fetch_full(blank, "H1")

    assert recorder.urls
    assert _inst_id_of(recorder.urls[0]) == "XAUUSD"


# ── 2) acquisition：失败原因必须逐条透传 ───────────────────────────────────


class _FailingFetcher(BaseFetcher):
    """恒定失败的假来源：``is_available()`` 为真，取数必抛指定原因。"""

    source_name = "FakeSource"

    def __init__(self, label: str, reason: str) -> None:
        self._label = label
        self._reason = reason

    def is_available(self) -> bool:
        return True

    def describe(self) -> str:
        return self._label

    def fetch_full(self, symbol: str, timeframe: str) -> pd.DataFrame:
        raise DataError(self._reason)

    def fetch_incremental(
        self, symbol: str, timeframe: str, since_ts: int
    ) -> pd.DataFrame:
        raise DataError(self._reason)


def _network_source_of(
    monkeypatch: pytest.MonkeyPatch, fakes: list[BaseFetcher]
) -> NetworkSource:
    """构造只含给定假来源、且不带通用 HTTP 接口的 ``NetworkSource``。

    用 monkeypatch 顶掉 ``list_network_sources``，避免构造期对真实主机做
    TCP 可达性探测（既慢又依赖网络）。
    """
    monkeypatch.setattr(
        "miaosuan.data.acquisition.list_network_sources",
        lambda *args, **kwargs: list(fakes),
    )
    source = NetworkSource()
    source._base_url = None  # 关掉通用 HTTP 分支，只测网络 fetcher 聚合
    source._sources = list(fakes)
    return source


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda ns: ns.fetch_full("XAUUSD", "H1"), id="fetch_full"),
        pytest.param(
            lambda ns: ns.fetch_incremental("XAUUSD", "H1", 0), id="fetch_incremental"
        ),
    ],
)
def test_acquisition_reports_every_source_failure(
    monkeypatch: pytest.MonkeyPatch, call
) -> None:
    """每个来源的失败原因都要能被区分出来：既进消息，也进 context["failures"]。"""
    fakes = [
        _FailingFetcher("FakeA", "boom-A"),
        _FailingFetcher("FakeB", "boom-B"),
    ]
    source = _network_source_of(monkeypatch, fakes)

    with pytest.raises(DataError) as excinfo:
        call(source)

    message = str(excinfo.value)
    failures = excinfo.value.context["failures"]

    assert len(failures) == 2, f"两个来源各失败一次，实际: {failures}"
    assert any("FakeA" in f and "boom-A" in f for f in failures)
    assert any("FakeB" in f and "boom-B" in f for f in failures)
    # 错误消息本身也必须带上，否则用户只看到一句「无可用来源」
    assert "boom-A" in message
    assert "boom-B" in message
