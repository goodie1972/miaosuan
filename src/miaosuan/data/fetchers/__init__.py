# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for full license text.

"""数据获取器包（``data/fetchers/``）。

集中收纳所有具体数据来源的 :class:`BaseFetcher` 实现：妙算本地库、TradingView、
akshare、tqsdk、OKX、Dukascopy。``acquisition.py`` 通过本包接入。

依赖方向（架构 §9.7 铁律）：本包**不**读取环境变量、不 import ``config``；
所有外部配置（代理、凭据）由调用方经构造参数注入。``list_network_sources()``
在**函数内**惰性构造各 fetcher，避免 import 期副作用。
"""

from __future__ import annotations

from .akshare_src import AkshareFetcher
from .base import BaseFetcher
from .binance import BinanceFetcher
from .dukascopy import DukascopyFetcher
from .mt4_bridge import DEFAULT_MT4_PORT, MT4BridgeFetcher
from .okx import OkxFetcher
from .shenji_db import ShenjiDBFetcher
from .tqsdk_src import TqsdkFetcher
from .tradingview import TradingViewFetcher

__all__ = [
    "BaseFetcher",
    "ShenjiDBFetcher",
    "TradingViewFetcher",
    "AkshareFetcher",
    "TqsdkFetcher",
    "OkxFetcher",
    "BinanceFetcher",
    "DukascopyFetcher",
    "MT4BridgeFetcher",
    "all_network_sources",
    "list_network_sources",
]


def all_network_sources(
    dukascopy_user: str | None = None,
    dukascopy_password: str | None = None,
    mt4_bridge_host: str | None = None,
    mt4_bridge_port: int | None = None,
    mt4_time_base: str | None = None,
) -> list[BaseFetcher]:
    """枚举所有网络来源 fetcher，**不做任何可用性探测**（立即返回）。

    与 :func:`list_network_sources` 的区别：后者会逐个 ``is_available()``
    过滤，其中 Binance / Dukascopy 走 TCP 连通性探测，主机不可达时**各卡满
    3s 超时**（实测 6s）。列来源只需枚举、不必当场判定可用性，故走本函数，
    把探测推迟到后台线程（见 ``acquisition._probe_in_background``）。

    MT4 Bridge 是**本机**来源（localhost TCP 连本机 MT4 终端挂的 EA，不耗外网），
    但只有显式传入 ``mt4_bridge_host`` 才纳入——没装 MT4 的机器上不应出现一个
    注定不可用的来源。

    Args:
        dukascopy_user: Dukascopy 实盘用户名（可选，预留；默认 ``None``）。
        dukascopy_password: Dukascopy 实盘密码（可选；默认 ``None``）。
        mt4_bridge_host: MT4 Bridge 监听地址（如 ``127.0.0.1``）；
            ``None`` 或空串 = 不启用 MT4 来源。
        mt4_bridge_port: MT4 Bridge 监听端口；默认 :data:`DEFAULT_MT4_PORT`。
        mt4_time_base: MT4 输出时间戳口径（``utc`` / ``broker`` / ``shanghai``）。

    Returns:
        全部网络来源 fetcher（按实现顺序），**未**过滤可用性。
    """
    sources: list[BaseFetcher] = [
        TradingViewFetcher(),
        AkshareFetcher(),
        TqsdkFetcher(),
        OkxFetcher(),
        BinanceFetcher(),
        DukascopyFetcher(user=dukascopy_user, password=dukascopy_password),
    ]
    mt4_host = (mt4_bridge_host or "").strip()
    if mt4_host:
        sources.append(
            MT4BridgeFetcher(
                host=mt4_host,
                port=mt4_bridge_port or DEFAULT_MT4_PORT,
                time_base=mt4_time_base or "utc",
            )
        )
    return sources


def list_network_sources(
    dukascopy_user: str | None = None,
    dukascopy_password: str | None = None,
    mt4_bridge_host: str | None = None,
    mt4_bridge_port: int | None = None,
    mt4_time_base: str | None = None,
) -> list[BaseFetcher]:
    """枚举所有「网络来源」fetcher，仅返回当前环境可用的。

    ⚠ 会逐个 ``is_available()`` 探测，**可能很慢**（不可达主机卡满超时）；
    仅用于「真的要取数」的路径。只想列来源请用 :func:`all_network_sources`。

    在**函数内**惰性构造各 fetcher（不在 import 时），避免 import 期副作用；
    第三方依赖缺失的 fetcher 其 ``is_available()`` 会优雅返回 ``False`` 而被过滤。

    Args:
        dukascopy_user: Dukascopy 实盘用户名（可选，预留；默认 ``None``）。
        dukascopy_password: Dukascopy 实盘密码（可选；默认 ``None``）。
        mt4_bridge_host / mt4_bridge_port / mt4_time_base: 同
            :func:`all_network_sources`；host 为空则不纳入 MT4 来源。

    Returns:
        可用网络来源 fetcher 列表（按实现顺序）。
    """
    candidates = all_network_sources(
        dukascopy_user=dukascopy_user,
        dukascopy_password=dukascopy_password,
        mt4_bridge_host=mt4_bridge_host,
        mt4_bridge_port=mt4_bridge_port,
        mt4_time_base=mt4_time_base,
    )
    return [src for src in candidates if src.is_available()]
