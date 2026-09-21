# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for full license text.

"""Binance 数据源（公开 REST，基于标准库 ``urllib``，**无需 API Key / 第三方依赖**）。

与 :mod:`~miaosuan.data.fetchers.okx` 同风格：只用标准库，避免为单一交易所引入
``ccxt`` 这类重型依赖。请求 ``GET /api/v3/klines``，分页拉取至上限根数。

``is_available()`` 会做一次**轻量 TCP 连通性探测**（``api.binance.com:443``，默认 3s）：
Binance 在部分地区 / 网络下不可达，探测失败时优雅隐藏该源，避免向用户展示
一个点了必然报错的来源。
"""

from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

from ...errors import DataError
from .base import BaseFetcher

__all__ = ["BinanceFetcher"]

BINANCE_BASE = "https://api.binance.com"
#: 连通性探测主机（``api.binance.com``）。
BINANCE_HOST = "api.binance.com"

#: 妙算周期 -> Binance interval 字符串。
_BINANCE_INTERVAL = {
    "M1": "1m",
    "M5": "5m",
    "M15": "15m",
    "M30": "30m",
    "H1": "1h",
    "H4": "4h",
    "D1": "1d",
    "W1": "1w",
}
#: 单次最多返回根数（Binance 公开接口上限 1000）。
_BINANCE_PAGE = 1000
#: 全量最多拉取根数（分页上限，防失控）。
_BINANCE_MAX_BARS = 5000


class BinanceFetcher(BaseFetcher):
    """Binance 公开行情读取器（标准库 ``urllib``，**无需 API Key**）。

    请求 ``GET /api/v3/klines?symbol=...&interval=...``，解析返回的二维数组
    （``[open_time_ms, open, high, low, close, volume, close_time_ms, ...]``），
    以 ``startTime = 上一页末根 close_time + 1`` 翻页，至多 :data:`_BINANCE_MAX_BARS`
    根；按 ``time`` 升序、去重，并**剔除尚未收盘的 forming 根**（close_time 晚于当前时间）。

    默认 ``symbol`` 为 ``BTCUSDT``；调用方传入非空 ``symbol`` 时优先使用（并大写规整）。
    """

    source_name = "Binance"

    def __init__(
        self,
        symbol: str | None = None,
        timeout: int = 20,
        max_bars: int = _BINANCE_MAX_BARS,
        probe_timeout: int = 3,
    ) -> None:
        """初始化。

        Args:
            symbol: 默认 Binance 交易对（如 ``BTCUSDT``）；为 ``None`` 时取 ``BTCUSDT``。
            timeout: HTTP 请求超时（秒）。
            max_bars: 全量最多拉取根数。
            probe_timeout: 连通性探测超时（秒）。
        """
        self._default_symbol = (symbol or "BTCUSDT").upper()
        self._timeout = timeout
        self._max_bars = max_bars
        self._probe_timeout = probe_timeout
        #: 连通性探测结果缓存（探测有网络往返，不可达时每次都要等满超时）。
        self._avail_cache: bool | None = None

    def is_available(self) -> bool:
        """公开接口 + 主机可达性探测（Binance 在部分网络下不可达）。

        探测结果**按实例缓存**：数据源列表接口会频繁调用本方法，若不缓存，
        主机不可达时每次都要空等 ``probe_timeout`` 秒，导致下拉卡顿。
        网络环境变化需重启进程（或新建 fetcher 实例） refreshed。
        """
        if self._avail_cache is not None:
            return self._avail_cache
        ok = False
        # 只解析 IPv4 并连第一个地址：``create_connection(host, port)`` 会依次尝试
        # IPv6 / IPv4 两个地址族，主机不可达时每个都等满 timeout，实测耗时翻倍（6s）。
        try:
            infos = socket.getaddrinfo(
                BINANCE_HOST, 443, socket.AF_INET, socket.SOCK_STREAM
            )
        except Exception:
            infos = []
        for info in infos[:1]:
            try:
                sock = socket.create_connection(info[4], timeout=self._probe_timeout)
            except Exception:
                continue
            ok = True
            try:
                sock.close()
            except Exception:
                pass
            break
        self._avail_cache = ok
        return ok

    def describe(self) -> str:
        if self.is_available():
            return f"Binance: {self._default_symbol} (public REST, no key)"
        return "Binance: (主机不可达)"

    # ── 内部 ─────────────────────────────────────────────────────────────

    def _http_get(self, url: str) -> list[list]:
        req = urllib.request.Request(url, headers={"User-Agent": "MiaoSuan/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                body = resp.read()
        except urllib.error.HTTPError as exc:
            raise DataError(f"Binance HTTP {exc.code}: {url}") from exc
        except Exception as exc:
            raise DataError(f"Binance 请求失败: {url}（{exc}）") from exc
        try:
            data = json.loads(body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise DataError(f"Binance 返回非 JSON: {url}（{exc}）") from exc
        if not isinstance(data, list):
            # Binance 错误响应形如 {"code": ..., "msg": ...}
            raise DataError(f"Binance API 错误: {data}")
        return data

    def _fetch_raw(self, symbol: str, timeframe: str) -> list[list]:
        """分页拉取原始蜡烛数组。"""
        interval = _BINANCE_INTERVAL.get(timeframe.upper())
        if interval is None:
            raise DataError(f"Binance 不支持周期: {timeframe}")

        rows: list[list] = []
        seen: set[int] = set()
        start_ms: int | None = None
        while len(rows) < self._max_bars:
            params: dict[str, str] = {
                "symbol": symbol,
                "interval": interval,
                "limit": str(_BINANCE_PAGE),
            }
            if start_ms is not None:
                params["startTime"] = str(start_ms)
            url = BINANCE_BASE + "/api/v3/klines?" + urllib.parse.urlencode(params)
            data = self._http_get(url)
            if not data:
                break
            for item in data:
                open_ms = int(item[0])
                if open_ms in seen:
                    continue
                seen.add(open_ms)
                rows.append(item)
            # 升序返回：用本页最后一根的 close_time + 1 翻页
            start_ms = int(data[-1][6]) + 1
            if len(data) < _BINANCE_PAGE:
                break
        return rows[: self._max_bars]

    # ── 接口 ─────────────────────────────────────────────────────────────

    def fetch_full(self, symbol: str, timeframe: str) -> pd.DataFrame:
        code = (symbol or "").strip().upper() or self._default_symbol
        raw = self._fetch_raw(code, timeframe)
        if not raw:
            raise DataError(f"Binance 无数据：{code} {timeframe}")

        now_ms = int(time.time() * 1000)
        # 剔除未收盘的 forming 根（close_time 晚于当前时间）
        if len(raw) > 1 and int(raw[-1][6]) > now_ms:
            raw = raw[:-1]

        out = []
        for item in raw:
            out.append(
                {
                    "time": int(int(item[0]) // 1000),
                    "open": float(item[1]),
                    "high": float(item[2]),
                    "low": float(item[3]),
                    "close": float(item[4]),
                    "volume": float(item[5] or 0.0),
                }
            )
        return self._finalize(pd.DataFrame(out))
