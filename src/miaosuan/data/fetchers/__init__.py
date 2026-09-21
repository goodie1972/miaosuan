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
    "list_network_sources",
]


def list_network_sources(
    dukascopy_user: str | None = None,
    dukascopy_password: str | None = None,
) -> list[BaseFetcher]:
    """枚举所有「网络来源」fetcher，仅返回当前环境可用的。

    在**函数内**惰性构造各 fetcher（不在 import 时），避免 import 期副作用；
    第三方依赖缺失的 fetcher 其 ``is_available()`` 会优雅返回 ``False`` 而被过滤。

    Args:
        dukascopy_user: Dukascopy 实盘用户名（可选，预留；默认 ``None``）。
        dukascopy_password: Dukascopy 实盘密码（可选；默认 ``None``）。

    Returns:
        可用网络来源 fetcher 列表（按实现顺序）。
    """
    candidates: list[BaseFetcher] = [
        TradingViewFetcher(),
        AkshareFetcher(),
        TqsdkFetcher(),
        OkxFetcher(),
        BinanceFetcher(),
        DukascopyFetcher(user=dukascopy_user, password=dukascopy_password),
    ]
    return [src for src in candidates if src.is_available()]
