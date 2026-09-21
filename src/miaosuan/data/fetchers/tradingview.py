# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for full license text.

"""TradingView 数据源（基于 ``tvdatafeed``，匿名可用）。

国内机器常需代理才能连通 TradingView。本类实现**不读环境变量**的代理自动探测
（直连 -> 常见本地端口 CONNECT 隧道测试 -> Windows 系统代理注册表），也可经
构造参数 ``proxy`` 显式指定。代理凭据来自外部注入，符合依赖方向铁律。

惰性导入 ``tvdatafeed`` / ``websocket``，缺依赖时 ``is_available()`` 优雅返回
``False``，保证 ``acquisition.py`` 在缺依赖环境下仍可正常 import。
"""

from __future__ import annotations

import socket
import ssl

import pandas as pd

from ...errors import DataError
from .base import BaseFetcher

__all__ = ["TradingViewFetcher"]

#: 妙算周期 -> tvDatafeed.Interval 属性名。
_TV_INTERVAL = {
    "M1": "in_1_minute",
    "M5": "in_5_minute",
    "M15": "in_15_minute",
    "M30": "in_30_minute",
    "H1": "in_1_hour",
    "H4": "in_4_hour",
    "D1": "in_daily",
    "W1": "in_weekly",
}

#: 自动探测交易所前缀（symbol 未显式带 ``EXCHANGE:`` 时）。
_PROBE_EXCHANGES = ["", "FX_IDC", "OANDA", "TVC", "NASDAQ", "NYSE"]

#: TradingView 行情主机（用于连通性探测）。
TV_HOST = "data.tradingview.com"
TV_PORT = 443

#: 常见本地代理端口（兜底探测，不读环境变量）。
_PROXY_PORTS = (7890, 7897, 10809, 1080, 8888, 8118)


class TradingViewFetcher(BaseFetcher):
    """TradingView 行情读取器（基于 ``tvdatafeed``，匿名访问）。"""

    source_name = "TradingView"

    def __init__(
        self,
        proxy: str | None = None,
        timeout: int = 10,
        max_bars: int = 5000,
    ) -> None:
        """初始化。

        Args:
            proxy: 显式 HTTP 代理（如 ``http://127.0.0.1:7890``）；为 ``None`` 时自动探测。
            timeout: 连通性探测超时（秒）。
            max_bars: 全量拉取的最大根数。
        """
        self._proxy = proxy
        self._timeout = timeout
        self._max_bars = max_bars

    def is_available(self) -> bool:
        # 同 akshare：用 find_spec 而非真正 import——tvDatafeed 首次导入耗时数秒，
        # 会把「数据源列表」接口的首个请求拖慢；真正的导入延迟到 fetch_full 内。
        try:
            import importlib.util

            return importlib.util.find_spec("tvDatafeed") is not None
        except Exception:
            return False

    def describe(self) -> str:
        if self.is_available():
            return f"TradingView: tvdatafeed (max_bars={self._max_bars})"
        return "TradingView: (未安装 tvdatafeed)"

    # ── 代理探测（不读 env）──────────────────────────────────────────────

    @staticmethod
    def _normalize_proxy(raw: str) -> str | None:
        """把任意代理文本规整为 ``http://host:port`` 形式；非法返回 ``None``。"""
        raw = (raw or "").strip()
        if not raw:
            return None
        if "://" not in raw:
            raw = "http://" + raw
        from urllib.parse import urlparse

        p = urlparse(raw)
        if not p.hostname or not p.port:
            return None
        auth = ""
        if p.username:
            auth = p.username + (":" + (p.password or "")) + "@"
        return f"http://{auth}{p.hostname}:{p.port}"

    @staticmethod
    def _tls_ok(timeout: int) -> bool:
        """直连 TLS 握手探测：能否建立到 TradingView:443 的隧道。"""
        try:
            sock = socket.create_connection((TV_HOST, TV_PORT), timeout=timeout)
        except Exception:
            return False
        try:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            with ctx.wrap_socket(sock, server_hostname=TV_HOST):
                return True
        except Exception:
            return False
        finally:
            try:
                sock.close()
            except Exception:
                pass

    @staticmethod
    def _proxy_tunnel_ok(proxy: str, timeout: int) -> bool:
        """经 HTTP 代理发 CONNECT，测试能否打通到 TradingView:443 的隧道。"""
        from urllib.parse import urlparse

        p = urlparse(proxy)
        try:
            s = socket.create_connection((p.hostname, p.port), timeout=timeout)
        except Exception:
            return False
        try:
            req = (
                f"CONNECT {TV_HOST}:{TV_PORT} HTTP/1.1\r\n"
                f"Host: {TV_HOST}:{TV_PORT}\r\n\r\n"
            )
            s.sendall(req.encode())
            resp = s.recv(4096)
            head = resp.split(b"\r\n", 1)[0]
            return resp.startswith(b"HTTP/") and b" 200" in head
        except Exception:
            return False
        finally:
            try:
                s.close()
            except Exception:
                pass

    @classmethod
    def _system_proxy_candidates(cls) -> list[str]:
        """从 Windows 系统代理注册表读取代理（不读环境变量）。"""
        cands: list[str] = []
        try:
            import winreg  # type: ignore
        except Exception:
            return cands
        try:
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Internet Settings",
            )
            enabled, _ = winreg.QueryValueEx(key, "ProxyEnable")
            if enabled:
                server, _ = winreg.QueryValueEx(key, "ProxyServer")
                for part in (server or "").split(";"):
                    if "=" in part:
                        part = part.split("=", 1)[1]
                    v = cls._normalize_proxy(part)
                    if v:
                        cands.append(v)
        except Exception:
            pass
        return cands

    def _detect_proxy(self) -> str | None:
        """代理自动探测：显式 > 直连可达 > 本地端口 > 系统代理。"""
        if self._proxy:
            return self._proxy
        if self._tls_ok(self._timeout):
            return None
        candidates: list[str] = [
            f"http://127.0.0.1:{port}" for port in _PROXY_PORTS
        ]
        candidates.extend(self._system_proxy_candidates())
        for cand in candidates:
            if self._proxy_tunnel_ok(cand, self._timeout):
                return cand
        return None

    @staticmethod
    def _apply_proxy_patch(proxy: str) -> None:
        """给 ``websocket.create_connection`` 注入代理参数（首次 get_hist 前调用）。"""
        from urllib.parse import urlparse

        p = urlparse(proxy)
        ph, pp = p.hostname, p.port
        auth = (p.username, p.password or "") if p.username else None
        try:
            import websocket  # websocket-client
        except Exception:
            return
        orig = websocket.create_connection

        def patched(url, *args, **kw):
            kw.setdefault("http_proxy_host", ph)
            kw.setdefault("http_proxy_port", pp)
            if auth:
                kw.setdefault("http_proxy_auth", auth)
            return orig(url, *args, **kw)

        try:
            websocket.create_connection = patched
        except Exception:
            pass

    # ── 拉取 ─────────────────────────────────────────────────────────────

    @staticmethod
    def _import_tv():
        try:
            from tvDatafeed import Interval, TvDatafeed
        except ImportError as exc:
            raise DataError(
                "未安装 tvdatafeed，请先安装："
                "pip install git+https://github.com/rongardF/tvdatafeed.git"
            ) from exc
        return TvDatafeed, Interval

    @staticmethod
    def _split_symbol(symbol: str) -> tuple[str, str | None]:
        """拆分 ``EXCHANGE:CODE`` 形式；无前缀返回 ``(code, None)``。"""
        s = (symbol or "").strip()
        if ":" in s:
            ex, code = s.split(":", 1)
            return code.strip(), ex.strip().upper()
        return s, None

    def fetch_full(self, symbol: str, timeframe: str) -> pd.DataFrame:
        if not self.is_available():
            raise DataError("TradingView 不可用：tvdatafeed 未安装")
        interval_name = _TV_INTERVAL.get(timeframe.upper())
        if interval_name is None:
            raise DataError(f"TradingView 不支持周期: {timeframe}")
        TvDatafeed, Interval = self._import_tv()

        proxy = self._detect_proxy()
        if proxy:
            self._apply_proxy_patch(proxy)

        try:
            feed = TvDatafeed()
        except Exception as exc:
            raise DataError(f"TradingView 连接失败: {exc}") from exc

        code, explicit_ex = self._split_symbol(symbol)
        interval = getattr(Interval, interval_name)
        probe = [explicit_ex] + _PROBE_EXCHANGES if explicit_ex else _PROBE_EXCHANGES

        df_raw = None
        for ex in probe:
            try:
                df_raw = feed.get_hist(
                    symbol=code, exchange=ex, interval=interval, n_bars=self._max_bars
                )
            except Exception:
                df_raw = None
            if df_raw is not None and not df_raw.empty:
                break

        if df_raw is None or df_raw.empty:
            raise DataError(
                f"TradingView 无数据：{symbol}（可用 EXCHANGE:CODE 指定交易所）"
            )

        out = []
        for row in df_raw.itertuples(index=True):
            idx = getattr(row, "Index")
            ts = int(idx.timestamp()) if hasattr(idx, "timestamp") else 0
            out.append(
                {
                    "time": ts,
                    "open": float(row.open),
                    "high": float(row.high),
                    "low": float(row.low),
                    "close": float(row.close),
                    "volume": float(getattr(row, "volume", 0.0) or 0.0),
                }
            )
        return self._finalize(pd.DataFrame(out))
