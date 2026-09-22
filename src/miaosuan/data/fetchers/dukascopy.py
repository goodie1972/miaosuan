# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for full license text.

"""Dukascopy 数据源（公开历史 K 线 JSON，基于标准库 ``urllib``，**无需第三方依赖**）。"""

from __future__ import annotations

import time
import urllib.parse
from typing import Any

import pandas as pd

from ...errors import DataError
from .base import BaseFetcher

__all__ = ["DukascopyFetcher"]

#: Dukascopy 公开历史 REST 基址。
DUK_BASE = "https://freeserv.dukascopy.com/2.0/rest/ps/public/candles"

#: 连通性探测主机（与 :data:`DUK_BASE` 同域）。
DUKASCOPY_HOST = "freeserv.dukascopy.com"

#: 妙算周期 -> Dukascopy 蜡烛 width（秒）。
_DUK_WIDTH = {
    "M1": 60,
    "M5": 300,
    "M15": 900,
    "M30": 1800,
    "H1": 3600,
    "H4": 14400,
    "D1": 86400,
    "W1": 604800,
}
#: 单次窗口跨度（天），分页用。
_DUK_CHUNK_DAYS = 365
#: 全量最多拉取根数。
_DUK_MAX_BARS = 5000


class DukascopyFetcher(BaseFetcher):
    """Dukascopy 公开历史行情读取器（标准库 ``urllib``，无需第三方依赖）。

    请求 Dukascopy 公开历史蜡烛接口（无需登录），按 ``width`` 秒聚合，
    分页拉取最近至多 :data:`_DUK_MAX_BARS` 根。返回字段解析兼容两种形态：
    对象数组（``{time, open, high, low, close, volume}``）与二维数组。

    ``username`` / ``password`` 为**未来实盘**预留（默认 ``None``）；历史数据
    走公开接口、**无需凭据**——因此 Dukascopy 的模拟 / 实盘账户是否过期
    **不影响**本读取器的历史取数（实时流才需 jForex 账户，超出本实现范围）。

    ``is_available()`` 做一次 TCP 可达性探测（结果缓存）：历史接口虽公开免登录，
    但主机在部分网络下不可达，探测可避免展示一个点了必然失败的来源。
    """

    source_name = "Dukascopy"

    def __init__(
        self,
        user: str | None = None,
        password: str | None = None,
        window_days: int = 365 * 5,
        timeout: int = 20,
        max_bars: int = _DUK_MAX_BARS,
        probe_timeout: int = 3,
    ) -> None:
        self._username = user
        self._password = password
        self._window_days = window_days
        self._timeout = timeout
        self._max_bars = max_bars
        self._probe_timeout = probe_timeout
        #: 可达性探测结果缓存（不可达时每次探测都要等满超时）。
        self._avail_cache: bool | None = None

    def is_available(self) -> bool:
        """公开接口免登录，但仍需主机可达；探测结果按实例缓存。

        与 :class:`~miaosuan.data.fetchers.binance.BinanceFetcher` 同策略：
        只解析 IPv4 并连第一个地址——``create_connection(host, port)`` 会依次
        尝试 IPv6 / IPv4 两个地址族，主机不可达时耗时翻倍。
        """
        if self._avail_cache is not None:
            return self._avail_cache
        # 探测实现下沉到基类 _probe_host（只解析 IPv4、连首个地址、超时可控、
        # socket 必关闭），避免各数据源各写一份易漏关 socket 的副本。
        self._avail_cache = self._probe_host(
            DUKASCOPY_HOST, 443, float(self._probe_timeout)
        )
        return self._avail_cache

    def describe(self) -> str:
        if not self.is_available():
            return f"Dukascopy: 主机不可达（{DUKASCOPY_HOST}）"
        cred = " (live creds set)" if self._username else " (public history)"
        return f"Dukascopy: {self._username or 'anonymous'}{cred}"

    # ── 内部 ─────────────────────────────────────────────────────────────

    def _http_get(self, url: str) -> Any:
        """GET + JSON 解析；复用基类 :meth:`BaseFetcher._http_get_json`。

        保留本薄封装是为了让 ``_fetch_raw`` 的调用点保持单一、便于测试替换。
        """
        return self._http_get_json(url, timeout=self._timeout)

    @staticmethod
    def _symbol_to_duk(symbol: str) -> str:
        """把妙算品种转为 Dukascopy 路径风格（如 ``XAUUSD`` -> ``XAU/USD``）。"""
        s = (symbol or "").strip().upper()
        if "/" in s:
            return s
        if len(s) == 6:
            return s[:3] + "/" + s[3:]
        return s

    def _fetch_raw(self, symbol: str, width: int) -> list[dict]:
        duk_sym = self._symbol_to_duk(symbol)
        end_ms = int(time.time() * 1000)
        start_ms = end_ms - int(self._window_days) * 86400 * 1000
        chunk_ms = int(_DUK_CHUNK_DAYS) * 86400 * 1000
        rows: list[dict] = []
        cursor = start_ms
        while cursor < end_ms and len(rows) < self._max_bars:
            nxt = min(cursor + chunk_ms, end_ms)
            params = {
                "from": str(cursor),
                "to": str(nxt),
                "width": str(width),
            }
            url = DUK_BASE + "/" + duk_sym + "?" + urllib.parse.urlencode(params)
            try:
                data = self._http_get(url)
            except DataError as exc:
                # HTTP 失败**如实向上抛**，不再静默 break——否则网络不通会被
                # 误报成「无数据」，与「该窗口真的没行情」难以区分。
                raise DataError(
                    f"Dukascopy 拉取中断（窗口 {cursor}~{nxt}）: {exc}"
                ) from exc
            if not data:
                cursor = nxt + 1
                continue
            for item in data:
                if isinstance(item, dict):
                    rows.append(
                        {
                            "time": int(item.get("time", 0)) // 1000,
                            "open": float(item.get("open", 0.0)),
                            "high": float(item.get("high", 0.0)),
                            "low": float(item.get("low", 0.0)),
                            "close": float(item.get("close", 0.0)),
                            "volume": float(item.get("volume", 0.0) or 0.0),
                        }
                    )
                elif isinstance(item, (list, tuple)) and len(item) >= 6:
                    rows.append(
                        {
                            "time": int(item[0]) // 1000,
                            "open": float(item[1]),
                            "high": float(item[2]),
                            "low": float(item[3]),
                            "close": float(item[4]),
                            "volume": float(item[5] or 0.0),
                        }
                    )
            cursor = nxt + 1
        return rows

    # ── 接口 ─────────────────────────────────────────────────────────────

    def fetch_full(self, symbol: str, timeframe: str) -> pd.DataFrame:
        width = _DUK_WIDTH.get(timeframe.upper())
        if width is None:
            raise DataError(f"Dukascopy 不支持周期: {timeframe}")
        raw = self._fetch_raw(symbol, width)
        if not raw:
            raise DataError(f"Dukascopy 无数据：{symbol} {timeframe}")
        df = pd.DataFrame(raw)
        if "time" not in df.columns:
            raise DataError(f"Dukascopy 返回数据无 time 字段: {symbol}")
        return self._finalize(df)
