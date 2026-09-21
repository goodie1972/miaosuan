# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for full license text.

"""OKX 数据源（公开 REST，基于标准库 ``urllib``，**无需任何第三方依赖**）。"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

from ...errors import DataError
from .base import BaseFetcher

__all__ = ["OkxFetcher"]

OKX_BASE = "https://www.okx.com"

#: 妙算周期 -> OKX bar 字符串。
_OKX_BAR = {
    "M1": "1m",
    "M5": "5m",
    "M15": "15m",
    "M30": "30m",
    "H1": "1H",
    "H4": "4H",
    "D1": "1D",
    "W1": "1W",
}
#: 单次最多返回根数（OKX 公开接口上限）。
_OKX_PAGE = 100
#: 全量最多拉取根数（分页上限，防失控）。
_OKX_MAX_BARS = 2000


class OkxFetcher(BaseFetcher):
    """OKX 公开行情读取器（基于标准库 ``urllib``，**无需任何第三方依赖**）。

    请求 ``GET /api/v5/market/candles?instId=...&bar=...``，解析返回的
    ``data`` 数组（``[ts_ms, open, high, low, close, vol, ...]``），分页拉取
    至多 :data:`_OKX_MAX_BARS` 根，按 ``time`` 升序、剔除未收盘的 forming 根。

    ``instId`` 默认 ``XAUUSD``（调用方可经 ``inst_id`` 覆盖为 OKX 实际标的，
    如 ``XAU-USDT`` 永续）。公开接口可用，``is_available()`` 恒为 ``True``。
    """

    source_name = "OKX"

    def __init__(
        self,
        inst_id: str | None = None,
        timeout: int = 20,
        max_bars: int = _OKX_MAX_BARS,
    ) -> None:
        self._inst_id = inst_id or "XAUUSD"
        self._timeout = timeout
        self._max_bars = max_bars

    def is_available(self) -> bool:
        # 公开 REST 接口，标准库即可，恒可用
        return True

    def describe(self) -> str:
        return f"OKX: {self._inst_id} (public REST)"

    # ── 内部 ─────────────────────────────────────────────────────────────

    def _http_get(self, url: str) -> dict:
        req = urllib.request.Request(url, headers={"User-Agent": "MiaoSuan/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                body = resp.read()
        except Exception as exc:
            raise DataError(f"OKX 请求失败: {url}（{exc}）") from exc
        try:
            return json.loads(body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise DataError(f"OKX 返回非 JSON: {url}（{exc}）") from exc

    def _fetch_raw(self, timeframe: str) -> list[list]:
        """分页拉取原始蜡烛数组（每元素 ``[ts_ms, o, h, l, c, vol, ...]``）。"""
        bar = _OKX_BAR.get(timeframe.upper())
        if bar is None:
            raise DataError(f"OKX 不支持周期: {timeframe}")
        rows: list[list] = []
        seen: set[int] = set()
        before: int | None = None
        while len(rows) < self._max_bars:
            params: dict[str, str] = {
                "instId": self._inst_id,
                "bar": bar,
                "limit": str(_OKX_PAGE),
            }
            if before is not None:
                params["before"] = str(before)
            url = OKX_BASE + "/api/v5/market/candles?" + urllib.parse.urlencode(params)
            body = self._http_get(url)
            if body.get("code") != "0":
                raise DataError(
                    f"OKX API 错误 {body.get('code')}: {body.get('msg')}"
                )
            data = body.get("data") or []
            if not data:
                break
            for item in data:
                ts_ms = int(item[0])
                if ts_ms in seen:
                    continue
                seen.add(ts_ms)
                rows.append(item)
            # OKX 降序返回；用本页最旧一根的 ts-1 翻页
            before = int(data[-1][0]) - 1
            if len(data) < _OKX_PAGE:
                break
        return rows

    # ── 接口 ─────────────────────────────────────────────────────────────

    def fetch_full(self, symbol: str, timeframe: str) -> pd.DataFrame:
        _ = symbol  # OKX 用 instId 而非 symbol，见 __init__
        raw = self._fetch_raw(timeframe)
        if not raw:
            raise DataError(f"OKX 无数据：{self._inst_id} {timeframe}")
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
        df = self._finalize(pd.DataFrame(out))
        # 剔除未收盘的 forming 根（最新一根 confirm == '0'）
        if len(df) > 0 and len(raw) > 0 and len(raw[0]) > 8:
            if str(raw[0][8]) == "0":
                df = df.iloc[:-1].reset_index(drop=True)
        return df
