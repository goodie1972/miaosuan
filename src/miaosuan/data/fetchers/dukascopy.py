# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for full license text.

"""Dukascopy 数据源（公开历史 K 线 JSON，基于标准库 ``urllib``，**无需第三方依赖**）。

使用 Dukascopy 新版公开 API (jetta.dukascopy.com/v1/candles/...) 获取历史蜡烛数据。
支持周期：M1, M5, M15, M30, H1, H4, D1, W1（月线 MN1 暂不支持）。
数据最早可追溯：XAUUSD 分钟级 2003-05-05，日线 1999-06-03。

连通性探测使用 TCP 连接（兼容现有测试与探测契约），实际数据获取使用新版 HTTP API。
"""

from __future__ import annotations

import time
import urllib.parse
from datetime import datetime, timezone, timedelta
from typing import Any

import pandas as pd

from ...errors import DataError
from .base import BaseFetcher

__all__ = ["DukascopyFetcher"]

#: Dukascopy 新版公开历史 REST 基址。
DUK_BASE = "https://jetta.dukascopy.com/v1/candles"

#: 连通性探测主机（用于 TCP 探测，兼容测试契约）。
DUKASCOPY_HOST = "jetta.dukascopy.com"

#: 妙算周期 -> Dukascopy API 路径段。
_DUK_TIMEFRAME_MAP = {
    "M1": "minute",
    "M5": "minute",
    "M15": "minute",
    "M30": "minute",
    "H1": "hour",
    "H4": "hour",
    "D1": "day",
    "W1": "day",   # 周线用日线聚合（后端不直接提供周线）
    "MN1": "day",  # 月线用日线聚合
}

#: 周期对应的最小时间粒度（秒），用于过滤聚合。
_TF_SECONDS = {
    "M1": 60,
    "M5": 300,
    "M15": 900,
    "M30": 1800,
    "H1": 3600,
    "H4": 14400,
    "D1": 86400,
    "W1": 604800,
    "MN1": 2592000,
}

#: 全量最多拉取根数（按年请求，每年最多约 250 根日线，足够覆盖 20 年）。
_DUK_MAX_BARS = 5000


class DukascopyFetcher(BaseFetcher):
    """Dukascopy 公开历史行情读取器（标准库 ``urllib``，无需第三方依赖）。

    请求 Dukascopy 新版公开历史蜡烛接口（无需登录），按年/月/日分桶拉取。
    返回数据格式包含 timestamp, multiplier, shift, times, opens, highs, lows, closes, volumes。

    ``username`` / ``password`` 为**未来实盘**预留（默认 ``None``）；历史数据
    走公开接口、**无需凭据**——因此 Dukascopy 的模拟 / 实盘账户是否过期
    **不影响**本读取器的历史取数（实时流才需 jForex 账户，超出本实现范围）。

    ``is_available()`` 做一次 HTTP 可达性探测（结果缓存，走代理）：历史接口虽公开免登录，
    但主机在部分网络下不可达，探测可避免展示一个点了必然失败的来源。

    代理支持：通过 ``proxy`` 参数显式指定（如 ``http://127.0.0.1:10808`` 或 ``socks5://127.0.0.1:10808``）；
    为 ``None`` 时自动读取环境变量 ``HTTP_PROXY``/``HTTPS_PROXY``/``ALL_PROXY``。
    """

    source_name = "Dukascopy"

    def __init__(
        self,
        user: str | None = None,
        password: str | None = None,
        window_days: int = 365 * 20,  # 默认拉取 20 年
        timeout: int = 60,
        max_bars: int = _DUK_MAX_BARS,
        probe_timeout: int = 10,
        proxy: str | None = None,
    ) -> None:
        self._username = user
        self._password = password
        self._window_days = window_days
        self._timeout = timeout
        self._max_bars = max_bars
        self._probe_timeout = probe_timeout
        #: 代理（HTTP/HTTPS），显式传入优先，否则由 config 统一提供
        self._proxy = proxy
        #: 可达性探测结果缓存（不可达时每次探测都要等满超时）。
        self._avail_cache: bool | None = None

    def _build_opener(self):
        """构建带代理的 opener。"""
        import urllib.request
        proxy = self._proxy
        if proxy:
            handler = urllib.request.ProxyHandler({"http": proxy, "https": proxy})
            return urllib.request.build_opener(handler)
        return urllib.request.build_opener()

    def is_available(self) -> bool:
        """公开接口免登录，但仍需主机可达；探测结果按实例缓存。

        使用基类 ``_probe_host`` 做 TCP 探测（仅 IPv4、连首个地址、超时可控、
        socket 必关闭），与 BinanceFetcher 等其他网络数据源保持一致。
        结果缓存避免重复探测导致超时等待。
        """
        if self._avail_cache is not None:
            return self._avail_cache
        self._avail_cache = self._probe_host(
            DUKASCOPY_HOST, 443, float(self._probe_timeout)
        )
        return self._avail_cache

    def describe(self) -> str:
        if not self.is_available():
            return f"Dukascopy: 主机不可达（{DUKASCOPY_HOST}）"
        cred = " (live creds set)" if self._username else " (public history)"
        return f"Dukascopy: {self._username or 'anonymous'}{cred}"

    def supported_timeframes(self) -> list[str]:
        """返回 Dukascopy 支持的周期列表。"""
        return list(_DUK_TIMEFRAME_MAP.keys())

    # ── 内部工具 ─────────────────────────────────────────────────────────────

    def _http_get(self, url: str) -> Any:
        """GET + JSON 解析；使用带代理的 opener，带重试。"""
        import urllib.request
        import time
        opener = self._build_opener()
        req = urllib.request.Request(url, headers={"User-Agent": "MiaoSuan/1.0"})
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with opener.open(req, timeout=self._timeout) as resp:
                    body = resp.read()
                import json
                try:
                    return json.loads(body.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    raise DataError(f"HTTP 返回非 JSON: {url}（{exc}）") from exc
            except Exception as exc:
                if attempt == max_retries - 1:
                    raise DataError(f"HTTP GET 失败 (重试 {max_retries} 次后): {url}（{exc}）") from exc
                time.sleep(1 * (attempt + 1))  # 1s, 2s, 3s

    @staticmethod
    def _symbol_to_duk(symbol: str) -> str:
        """把妙算品种转为 Dukascopy 路径风格（如 ``XAUUSD`` -> ``XAU-USD``）。"""
        s = (symbol or "").strip().upper()
        if "/" in s:
            return s.replace("/", "-")
        if "-" in s:
            return s
        if len(s) == 6:
            return s[:3] + "-" + s[3:]
        return s

    def _build_url(self, duk_symbol: str, tf_path: str, year: int, month: int | None = None, day: int | None = None) -> str:
        """构建 Dukascopy API URL。"""
        # 价格类型固定为 BID（买价），也可扩展为参数
        price_type = "BID"
        if tf_path == "day":
            return f"{DUK_BASE}/day/{duk_symbol}/{price_type}/{year}"
        elif tf_path == "hour":
            if month is None:
                raise ValueError("hour timeframe requires month")
            return f"{DUK_BASE}/hour/{duk_symbol}/{price_type}/{year}/{month}"
        elif tf_path == "minute":
            if month is None or day is None:
                raise ValueError("minute timeframe requires month and day")
            return f"{DUK_BASE}/minute/{duk_symbol}/{price_type}/{year}/{month}/{day}"
        else:
            raise ValueError(f"Unknown timeframe path: {tf_path}")

    def _parse_response(self, data: dict[str, Any], target_tf_seconds: int) -> list[dict[str, Any]]:
        """解析 Dukascopy API 返回的数据为统一格式（参考 dukascopy-node 实现）。

        数据格式说明：
        - timestamp: 基础时间戳（毫秒）
        - shift: 时间步长（毫秒）
        - times: 时间增量数组（每个值表示距离上一根 K 线的 shift 倍数，0 表示连续，>1 表示有间隙）
        - open/high/low/close: 基础价格值（已乘以 multiplier 的整数单位）
        - opens/highs/lows/closes: 价格增量数组（相对于前一根 K 线的变化量，已乘以 multiplier）
        - volumes: 成交量
        - multiplier: 价格乘数（如 0.001）
        
        返回格式：list of {time, open, high, low, close, volume}
        time 为 Unix 秒级时间戳（UTC）。
        """
        if not data:
            return []

        base_ts = data.get("timestamp", 0)  # 毫秒
        multiplier = data.get("multiplier", 1.0)
        shift = data.get("shift", 0)  # 毫秒
        times = data.get("times", [])
        opens_delta = data.get("opens", [])
        highs_delta = data.get("highs", [])
        lows_delta = data.get("lows", [])
        closes_delta = data.get("closes", [])
        volumes = data.get("volumes", [])

        if not times:
            return []

        # 基础价格单位（整数，已乘以 multiplier）
        open_units = round(data.get("open", 0) / multiplier) if multiplier else 0
        high_units = round(data.get("high", 0) / multiplier) if multiplier else 0
        low_units = round(data.get("low", 0) / multiplier) if multiplier else 0
        close_units = round(data.get("close", 0) / multiplier) if multiplier else 0

        rows = []
        timestamp_ms = base_ts
        prev_close_units = close_units

        for i, time_delta in enumerate(times):
            # 处理时间间隙：time_delta > 1 表示缺失了 (time_delta - 1) 根 K 线
            # 第一个元素 time_delta=0 表示无间隙
            gap = time_delta - (0 if i == 0 else 1)
            for g in range(gap):
                # 填充平坦 K 线（价格 = 前一根收盘价，成交量 = 0）
                flat_ts = timestamp_ms + (g + (0 if i == 0 else 1)) * shift
                flat_price = prev_close_units * multiplier
                rows.append({
                    "time": int(flat_ts // 1000),
                    "open": float(flat_price),
                    "high": float(flat_price),
                    "low": float(flat_price),
                    "close": float(flat_price),
                    "volume": 0.0,
                })

            # 当前根 K 线的时间
            timestamp_ms += time_delta * shift

            # 累加价格增量
            if i < len(opens_delta):
                open_units += opens_delta[i]
            if i < len(highs_delta):
                high_units += highs_delta[i]
            if i < len(lows_delta):
                low_units += lows_delta[i]
            if i < len(closes_delta):
                close_units += closes_delta[i]

            prev_close_units = close_units

            if i < len(opens_delta) and i < len(highs_delta) and i < len(lows_delta) and i < len(closes_delta):
                o = open_units * multiplier
                h = high_units * multiplier
                l = low_units * multiplier
                c = close_units * multiplier
                v = volumes[i] if i < len(volumes) else 0.0
                
                # 过滤周期不匹配的数据（如从分钟线聚合 M5/M15/M30/H1/H4）
                if target_tf_seconds > 60:
                    bar_dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
                    if target_tf_seconds == 300:  # M5
                        if bar_dt.minute % 5 != 0 or bar_dt.second != 0:
                            continue
                    elif target_tf_seconds == 900:  # M15
                        if bar_dt.minute % 15 != 0 or bar_dt.second != 0:
                            continue
                    elif target_tf_seconds == 1800:  # M30
                        if bar_dt.minute % 30 != 0 or bar_dt.second != 0:
                            continue
                    elif target_tf_seconds == 3600:  # H1
                        if bar_dt.minute != 0 or bar_dt.second != 0:
                            continue
                    elif target_tf_seconds == 14400:  # H4
                        if bar_dt.hour % 4 != 0 or bar_dt.minute != 0 or bar_dt.second != 0:
                            continue

                rows.append({
                    "time": int(timestamp_ms // 1000),
                    "open": float(o),
                    "high": float(h),
                    "low": float(l),
                    "close": float(c),
                    "volume": float(v or 0.0),
                })
        return rows

    def _fetch_timeframe(self, duk_symbol: str, tf: str, years: list[int]) -> list[dict[str, Any]]:
        """按年/月/日拉取指定周期的数据。"""
        tf_path = _DUK_TIMEFRAME_MAP[tf]
        target_seconds = _TF_SECONDS[tf]
        all_rows = []

        # 获取时间范围
        start_dt = getattr(self, '_fetch_start_dt', None)
        end_dt = getattr(self, '_fetch_end_dt', None)
        fetch_tf = getattr(self, '_fetch_tf', tf)

        for year in years:
            # 检查年份是否在范围内
            if start_dt and year < start_dt.year:
                continue
            if end_dt and year > end_dt.year:
                continue

            if tf_path == "day":
                # 日线：按年请求
                url = self._build_url(duk_symbol, "day", year)
                try:
                    data = self._http_get(url)
                    rows = self._parse_response(data, target_seconds)
                    # 过滤时间范围外的数据
                    if start_dt:
                        rows = [r for r in rows if r["time"] >= int(start_dt.timestamp())]
                    if end_dt:
                        rows = [r for r in rows if r["time"] <= int(end_dt.timestamp())]
                    all_rows.extend(rows)
                except DataError as exc:
                    # 400 通常表示该年份数据尚不可用（如未来年份），跳过而非报错
                    if "400" in str(exc) or "timeout" in str(exc).lower() or "timed out" in str(exc).lower():
                        print(f"  跳过年份 {year}: {exc}")
                        continue
                    raise DataError(f"Dukascopy 拉取中断（窗口 {duk_symbol} {tf} {year}）: {exc}") from exc

            elif tf_path == "hour":
                # 小时线：按月请求
                for month in range(1, 13):
                    # 检查月份是否在范围内
                    if start_dt and (year < start_dt.year or (year == start_dt.year and month < start_dt.month)):
                        continue
                    if end_dt and (year > end_dt.year or (year == end_dt.year and month > end_dt.month)):
                        continue
                    url = self._build_url(duk_symbol, "hour", year, month)
                    try:
                        data = self._http_get(url)
                        rows = self._parse_response(data, target_seconds)
                        if start_dt:
                            rows = [r for r in rows if r["time"] >= int(start_dt.timestamp())]
                        if end_dt:
                            rows = [r for r in rows if r["time"] <= int(end_dt.timestamp())]
                        all_rows.extend(rows)
                    except DataError as exc:
                        if "400" in str(exc) or "timeout" in str(exc).lower() or "timed out" in str(exc).lower():
                            print(f"  跳过月份 {year}-{month:02d}: {exc}")
                            continue
                        raise DataError(f"Dukascopy 拉取中断（窗口 {duk_symbol} {tf} {year}-{month:02d}）: {exc}") from exc

            elif tf_path == "minute":
                # 分钟线：按日请求
                import calendar
                for month in range(1, 13):
                    # 检查月份是否在范围内
                    if start_dt and (year < start_dt.year or (year == start_dt.year and month < start_dt.month)):
                        continue
                    if end_dt and (year > end_dt.year or (year == end_dt.year and month > end_dt.month)):
                        continue
                    days_in_month = calendar.monthrange(year, month)[1]
                    for day in range(1, days_in_month + 1):
                        # 检查日期是否在范围内
                        try:
                            dt = datetime(year, month, day, tzinfo=timezone.utc)
                            if dt < start_dt or dt > end_dt:
                                continue
                            if dt.weekday() >= 5:  # 周六日
                                continue
                        except ValueError:
                            continue
                        url = self._build_url(duk_symbol, "minute", year, month, day)
                        try:
                            data = self._http_get(url)
                            rows = self._parse_response(data, target_seconds)
                            all_rows.extend(rows)
                        except DataError as exc:
                            if "400" in str(exc) or "timeout" in str(exc).lower() or "timed out" in str(exc).lower():
                                continue
                            raise DataError(f"Dukascopy 拉取中断（窗口 {duk_symbol} {tf} {year}-{month:02d}-{day:02d}）: {exc}") from exc

        return all_rows

    def _fetch_raw(self, symbol: str, width: int) -> list[dict[str, Any]]:
        """统一入口：根据周期拉取原始数据。"""
        duk_symbol = self._symbol_to_duk(symbol)
        # 找到对应的妙算周期
        tf = None
        for k, v in _TF_SECONDS.items():
            if v == width:
                tf = k
                break
        if tf is None:
            raise DataError(f"Dukascopy 不支持 width={width}")

        # 计算需要的日期范围
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=self._window_days)
        
        # 根据周期决定粒度
        if tf == "D1":
            years = list(range(start_dt.year, end_dt.year + 1))
        elif tf in ("H1", "H4"):
            # 只获取需要的年月
            years = list(range(start_dt.year, end_dt.year + 1))
        elif tf in ("M1", "M5", "M15", "M30"):
            years = list(range(start_dt.year, end_dt.year + 1))
        else:
            years = list(range(start_dt.year, end_dt.year + 1))

        # 将时间范围传递给 _fetch_timeframe（通过实例变量）
        self._fetch_start_dt = start_dt
        self._fetch_end_dt = end_dt
        self._fetch_tf = tf

        return self._fetch_timeframe(duk_symbol, tf, years)

    # ── 接口 ─────────────────────────────────────────────────────────────

    def fetch_full(self, symbol: str, timeframe: str) -> pd.DataFrame:
        tf = timeframe.upper()
        width = _TF_SECONDS.get(tf)
        if width is None:
            raise DataError(f"Dukascopy 不支持周期: {timeframe}")
        raw = self._fetch_raw(symbol, width)
        if not raw:
            raise DataError(f"Dukascopy 无数据：{symbol} {timeframe}")
        df = pd.DataFrame(raw)
        if "time" not in df.columns:
            raise DataError(f"Dukascopy 返回数据无 time 字段: {symbol}")
        return self._finalize(df)

    