# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for full license text.

"""数据获取器包（``data/fetchers/``）。

集中收纳所有具体数据来源的 :class:`BaseFetcher` 实现：神机本地库、TradingView、
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
from .okx import OkxFetcher
from .shenji_db import ShenjiDBFetcher
from .tradingview import TradingViewFetcher
from .tqsdk_src import TqsdkFetcher

__all__ = [
    "BaseFetcher",
    "ShenjiDBFetcher",
    "TradingViewFetcher",
    "AkshareFetcher",
    "TqsdkFetcher",
    "OkxFetcher",
    "BinanceFetcher",
    "DukascopyFetcher",
    "all_network_sources",
    "list_network_sources",
]


def all_network_sources(
    dukascopy_user: str | None = None,
    dukascopy_password: str | None = None,
) -> list[BaseFetcher]:
    """枚举所有网络来源 fetcher，**不做任何可用性探测**（立即返回）。

    与 :func:`list_network_sources` 的区别：后者会逐个 ``is_available()``
    过滤，其中 Binance / Dukascopy 走 TCP 连通性探测，主机不可达时**各卡满
    3s 超时**（实测 6s）。列来源只需枚举、不必当场判定可用性，故走本函数，
    把探测推迟到后台线程（见 ``acquisition._probe_in_background``）。

    Args:
        dukascopy_user: Dukascopy 实盘用户名（可选，预留；默认 ``None``）。
        dukascopy_password: Dukascopy 实盘密码（可选；默认 ``None``）。

    Returns:
        全部网络来源 fetcher（按实现顺序），**未**过滤可用性。
    """
    return [
        TradingViewFetcher(),
        AkshareFetcher(),
        TqsdkFetcher(),
        OkxFetcher(),
        BinanceFetcher(),
        DukascopyFetcher(user=dukascopy_user, password=dukascopy_password),
    ]


def list_network_sources(
    dukascopy_user: str | None = None,
    dukascopy_password: str | None = None,
) -> list[BaseFetcher]:
    """枚举所有「网络来源」fetcher，仅返回当前环境可用的。

    ⚠ 会逐个 ``is_available()`` 探测，**可能很慢**（不可达主机卡满超时）；
    仅用于「真的要取数」的路径。只想列来源请用 :func:`all_network_sources`。

    在**函数内**惰性构造各 fetcher（不在 import 时），避免 import 期副作用；
    第三方依赖缺失的 fetcher 其 ``is_available()`` 会优雅返回 ``False`` 而被过滤。

    Args:
        dukascopy_user: Dukascopy 实盘用户名（可选，预留；默认 ``None``）。
        dukascopy_password: Dukascopy 实盘密码（可选；默认 ``None``）。

    Returns:
        可用网络来源 fetcher 列表（按实现顺序）。
    """
    candidates = all_network_sources(
        dukascopy_user=dukascopy_user, dukascopy_password=dukascopy_password
    )
    return [src for src in candidates if src.is_available()]
