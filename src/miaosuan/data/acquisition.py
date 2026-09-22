# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""统一数据获取模块（架构 §3.2）。

将两类数据来源——**神机本地数据库**与**网络下载**——统一封装在
``fetch(symbol, timeframe, since=None)`` 接口背后。上层（loader、CLI、
纸面交易）只调用 ``fetch()``，不感知来源细节。

增量优先策略：
* 本地已有缓存 → 只拉取 ``max(time)`` 之后的增量部分并追加；
* 本地无缓存（首跑）→ 自动退化为全量获取，拉取完整历史；
* 增量 / 全量的判定由模块内部自动完成，调用方无需感知。

职责边界：
* 本模块**只负责取数**——统一抽象来源、处理认证、缓存落地、增量 / 全量决策；
* **不**做特征计算、**不**碰策略逻辑、**不**写回神机；
* 取回的原始数据统一落盘为 parquet（沿用 ``loader`` 的 ``infer_symbol_timeframe``
  命名约定），后续由 ``loader.load()`` 加载为 ``Panel``。
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import data_acquisition_config
from .fetchers import BaseFetcher, ShenjiDBFetcher, all_network_sources, list_network_sources
from ..errors import DataError

__all__ = [
    "DataSource",
    "ShenjiLocalSource",
    "NetworkSource",
    "TypedNetworkSource",
    "GenericHttpSource",
    "DataAcquisition",
    "fetch",
]

#: 默认本地缓存目录列表（与 ``server.py`` 的 ``_DATA_DIRS`` 对齐）。
_DEFAULT_CACHE_DIRS: tuple[Path, ...] = (
    Path(r"D:\K线数据"),
    Path(__file__).resolve().parents[2] / "data",
)

#: 支持的 OHLCV 列名（与 ``loader.panel_from_frame`` 对齐）。
_REQUIRED_COLUMNS = ("time", "open", "high", "low", "close", "volume", "tick_volume")


def _slugify(text: str) -> str:
    """把备注转为安全的文件名片段（保留字母/数字/中文，其余替换为下划线，最长 60）。"""
    slug = re.sub(r"\W+", "_", (text or "").strip())
    return slug.strip("_")[:60]


#: 数据源类型（按「数据源类型」划分，而非其他维度）。
SOURCE_TYPE_SHENJI = "Shenji"
SOURCE_TYPE_TRADINGVIEW = "TradingView"
SOURCE_TYPE_OKX = "OKX"
SOURCE_TYPE_DUKASCOPY = "Dukascopy"
SOURCE_TYPE_BINANCE = "Binance"
SOURCE_TYPE_AKSHARE = "AkShare"
SOURCE_TYPE_OTHER = "其他"

#: fetcher 类名 → 数据源类型。新增 fetcher 只需在此登记。
_FETCHER_TYPE_MAP: dict[str, str] = {
    "ShenjiDBFetcher": SOURCE_TYPE_SHENJI,
    "TradingViewFetcher": SOURCE_TYPE_TRADINGVIEW,
    "OkxFetcher": SOURCE_TYPE_OKX,
    "DukascopyFetcher": SOURCE_TYPE_DUKASCOPY,
    "BinanceFetcher": SOURCE_TYPE_BINANCE,
    "AkshareFetcher": SOURCE_TYPE_AKSHARE,
    "TqsdkFetcher": SOURCE_TYPE_OTHER,
}


class DataSource(ABC):
    """数据来源抽象基类。

    子类实现 :meth:`fetch_full` 与 :meth:`fetch_incremental`，
    本基类负责数据合法性校验。
    """

    @abstractmethod
    def fetch_full(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """全量获取指定品种 + 周期的完整历史数据。

        Returns:
            含 ``time/open/high/low/close/(tick_)volume`` 列的 DataFrame。
        """
        ...

    @abstractmethod
    def fetch_incremental(
        self, symbol: str, timeframe: str, since_ts: int
    ) -> pd.DataFrame:
        """增量获取：只拉取 ``since_ts``（Unix 秒）之后的数据。

        Returns:
            同 :meth:`fetch_full`；可能为空 DataFrame（无新数据）。
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """该来源当前是否可用（配置了凭据 / 地址）。"""
        ...

    @abstractmethod
    def describe(self) -> str:
        """人类可读的来源描述（供 UI 展示）。"""
        ...

    @abstractmethod
    def source_type(self) -> str:
        """该来源的数据源类型标签（Shenji / TradingView / OKX / Dukascopy / 其他）。

        用于 UI 按「数据源类型」归类、并按市场画像严格过滤，而非按类实例名。
        """
        ...

    #: 可用性判定是否**瞬时可得**（本地文件 / 配置判定，无任何网络往返）。
    #: 为 ``True`` 时 :meth:`DataAcquisition.list_sources` 会同步判定并直接给出
    #: 真实值；网络来源保持 ``False``，走后台异步探测——否则请求线程会被不可达
    #: 主机（各卡满 3s 超时）拖住，切换画像就卡 6 秒。
    local_probe: bool = False

    # ── 内部工具 ───────────────────────────────────────────────────

    @staticmethod
    def _validate_df(df: pd.DataFrame, *, symbol: str, timeframe: str) -> pd.DataFrame:
        """校验从来源拿到的 DataFrame 是否包含必需列。"""
        if df is None or len(df) == 0:
            raise DataError(
                f"来源返回空数据: {symbol} {timeframe}",
                context={"symbol": symbol, "timeframe": timeframe},
            )
        if "tick_volume" not in df.columns and "volume" in df.columns:
            df = df.copy()
            df["tick_volume"] = df["volume"]
        missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise DataError(
                f"数据缺少必需列: {missing}",
                context={"symbol": symbol, "timeframe": timeframe, "missing": missing},
            )
        return df


class ShenjiLocalSource(DataSource):
    """神机本地数据库数据源（委托给 :class:`ShenjiDBFetcher`）。

    通过神机落库的只读 SQLite 读取行情。``db_path`` 解析优先级：
    构造参数 > ``DataAcquisitionConfig.shenji_db_path``（含默认候选路径）。
    实际读库逻辑全部在 :class:`~miaosuan.data.fetchers.ShenjiDBFetcher`，
    本类仅做配置解析与「不可用即报错」的兼容处理（保持 ``DataSource`` 契约）。
    """

    def __init__(self, db_path: str | None = None) -> None:
        cfg = data_acquisition_config()
        db_path = db_path or cfg.shenji_db_path or None
        self._fetcher = ShenjiDBFetcher(db_path) if db_path else None

    #: 可用性 = 本地 DB 文件是否存在（``os.path.isfile``），瞬时可得、无网络往返。
    local_probe = True

    def is_available(self) -> bool:
        return self._fetcher is not None and self._fetcher.is_available()

    def describe(self) -> str:
        if self._fetcher is not None:
            return self._fetcher.describe()
        return "Shenji DB: (未配置)"

    def source_type(self) -> str:
        return SOURCE_TYPE_SHENJI

    def fetch_full(self, symbol: str, timeframe: str) -> pd.DataFrame:
        if self._fetcher is None or not self._fetcher.is_available():
            raise DataError("神机本地数据库不可用：未配置 db_path")
        df = self._fetcher.fetch_full(symbol, timeframe)
        return self._validate_df(df, symbol=symbol, timeframe=timeframe)

    def fetch_incremental(
        self, symbol: str, timeframe: str, since_ts: int
    ) -> pd.DataFrame:
        if self._fetcher is None or not self._fetcher.is_available():
            raise DataError("神机本地数据库不可用：未配置 db_path")
        df = self._fetcher.fetch_incremental(symbol, timeframe, since_ts)
        if df is None or len(df) == 0:
            return pd.DataFrame()
        return self._validate_df(df, symbol=symbol, timeframe=timeframe)


class NetworkSource(DataSource):
    """网络下载数据源（聚合多个网络 fetcher）。

    优先级：
    1. 若配置了通用 HTTP JSON 接口（``MIAOSUAN_DATA_API_URL``），复用既有
       通用 HTTP 拉取逻辑；
    2. 否则（或通用接口失败）依次尝试 ``list_network_sources()`` 返回的可用
       网络 fetcher（TradingView / akshare / tqsdk / OKX / Dukascopy），返回
       第一个非空结果。

    所有配置来自 ``data_acquisition_config()``（唯一 env 边界），本类不读环境变量。
    """

    def __init__(
        self,
        base_url: str | None = None,
        timeout: int | None = None,
        dukascopy_user: str | None = None,
        dukascopy_password: str | None = None,
    ) -> None:
        cfg = data_acquisition_config()
        self._base_url: str | None = (
            base_url if base_url is not None else (cfg.data_api_url or None)
        )
        self._timeout: int = timeout if timeout is not None else cfg.timeout
        # 网络 fetcher 列表（惰性构造于 list_network_sources 内）
        self._sources: list[BaseFetcher] = list_network_sources(
            dukascopy_user=dukascopy_user or cfg.dukascopy_user or None,
            dukascopy_password=dukascopy_password or cfg.dukascopy_password or None,
        )

    def is_available(self) -> bool:
        return (self._base_url is not None and len(self._base_url) > 0) or any(
            s.is_available() for s in self._sources
        )

    def describe(self) -> str:
        parts: list[str] = []
        if self._base_url:
            parts.append(f"Network API: {self._base_url}")
        for s in self._sources:
            if s.is_available():
                parts.append(s.describe())
        if not parts:
            return "Network: (未配置)"
        return "Network: " + " | ".join(parts)

    def source_type(self) -> str:
        return SOURCE_TYPE_OTHER

    def fetch_full(self, symbol: str, timeframe: str) -> pd.DataFrame:
        # 逐个来源收集失败原因：早期版本 ``except Exception: pass/continue`` 把所有
        # 失败压成一句「无可用网络来源」，用户无法区分「没装包 / 网络不通 / 品种不支持」。
        failures: list[str] = []
        # 1) 通用 HTTP JSON 接口
        if self._base_url:
            try:
                df = self._generic_fetch(symbol, timeframe, since_ts=None)
                if df is not None and len(df) > 0:
                    return self._validate_df(df, symbol=symbol, timeframe=timeframe)
            except Exception as exc:
                failures.append(f"通用HTTP接口({self._base_url}): {exc}")
        # 2) 网络 fetcher 依次尝试
        for src in self._sources:
            if not src.is_available():
                continue
            try:
                df = src.fetch_full(symbol, timeframe)
                if df is not None and len(df) > 0:
                    return self._validate_df(df, symbol=symbol, timeframe=timeframe)
            except Exception as exc:
                failures.append(f"{src.describe()}: {exc}")
        return self._raise_no_source(symbol, timeframe, failures, incremental=False)

    def fetch_incremental(
        self, symbol: str, timeframe: str, since_ts: int
    ) -> pd.DataFrame:
        # 通用 HTTP 接口支持 since 参数；网络 fetcher 内部按 since 过滤
        failures: list[str] = []
        if self._base_url:
            try:
                df = self._generic_fetch(symbol, timeframe, since_ts=since_ts)
                if df is not None and len(df) > 0:
                    return self._validate_df(df, symbol=symbol, timeframe=timeframe)
            except Exception as exc:
                failures.append(f"通用HTTP接口({self._base_url}): {exc}")
        for src in self._sources:
            if not src.is_available():
                continue
            try:
                df = src.fetch_incremental(symbol, timeframe, since_ts)
                if df is not None and len(df) > 0:
                    return self._validate_df(df, symbol=symbol, timeframe=timeframe)
            except Exception as exc:
                failures.append(f"{src.describe()}: {exc}")
        return self._raise_no_source(symbol, timeframe, failures, incremental=True)

    def _raise_no_source(
        self,
        symbol: str,
        timeframe: str,
        failures: list[str],
        *,
        incremental: bool,
    ) -> pd.DataFrame:
        """抛出「无可用网络来源」并**带上每个来源的失败原因**。

        失败原因同时进错误消息与 ``context["failures"]``：前者让用户立刻看见，
        后者供 UI / 日志结构化消费。
        """
        kind = "无可用网络来源（增量）" if incremental else "无可用网络来源"
        detail = "；".join(failures) if failures else "无可用来源"
        raise DataError(
            f"{kind}: {symbol} {timeframe}（{detail}）",
            context={
                "sources": [s.describe() for s in self._sources],
                "failures": failures,
            },
        )

    # ── 通用 HTTP JSON 逻辑（复用既有实现）────────────────────────────────

    def _generic_fetch(
        self, symbol: str, timeframe: str, since_ts: int | None
    ) -> pd.DataFrame | None:
        import urllib.parse
        import urllib.request

        url = (
            self._base_url.rstrip("/")
            + f"/candles?symbol={urllib.parse.quote(symbol)}"
            + f"&timeframe={urllib.parse.quote(timeframe)}"
        )
        if since_ts is not None:
            url += f"&since={since_ts}"
        raw = self._http_get(url)
        return self._parse_json(raw)

    def _http_get(self, url: str) -> dict[str, Any]:
        import urllib.request

        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                body = resp.read()
        except Exception as exc:
            raise DataError(f"网络请求失败: {url}（{exc}）") from exc
        try:
            return json.loads(body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise DataError(f"网络返回非 JSON: {url}（{exc}）") from exc

    @staticmethod
    def _parse_json(raw: dict[str, Any]) -> pd.DataFrame:
        """把服务商返回的 JSON 转为 DataFrame。

        约定格式::

            {"candles": [{"time": ..., "open": ..., ...}, ...]}
        """
        candles = raw.get("candles") or raw.get("data") or []
        if not candles:
            return pd.DataFrame()
        return pd.DataFrame(candles)


class TypedNetworkSource(DataSource):
    """单一网络 fetcher 包装为**独立**、按类型可识别的数据源。

    每个网络 fetcher（TradingView / OKX / Dukascopy / Akshare / Tqsdk）都成为
    独立的数据源实例，其 ``source_type`` 由 :data:`_FETCHER_TYPE_MAP` 决定，
    便于 UI 按「数据源类型」归类，并依市场画像严格过滤。
    """

    def __init__(self, fetcher: BaseFetcher) -> None:
        self._fetcher = fetcher

    def source_type(self) -> str:
        return _FETCHER_TYPE_MAP.get(type(self._fetcher).__name__, SOURCE_TYPE_OTHER)

    def is_available(self) -> bool:
        return self._fetcher.is_available()

    def describe(self) -> str:
        return self._fetcher.describe()

    def fetch_full(self, symbol: str, timeframe: str) -> pd.DataFrame:
        df = self._fetcher.fetch_full(symbol, timeframe)
        return self._validate_df(df, symbol=symbol, timeframe=timeframe)

    def fetch_incremental(
        self, symbol: str, timeframe: str, since_ts: int
    ) -> pd.DataFrame:
        df = self._fetcher.fetch_incremental(symbol, timeframe, since_ts)
        if df is None or len(df) == 0:
            return pd.DataFrame()
        return self._validate_df(df, symbol=symbol, timeframe=timeframe)


class GenericHttpSource(DataSource):
    """通用 HTTP JSON 行情接口（归入「其他」类型）。

    仅当配置了 ``MIAOSUAN_DATA_API_URL`` 时可用；作为「其他」类别下的一个独立来源。
    """

    def __init__(self, base_url: str, timeout: int = 30) -> None:
        self._base_url = (base_url or "").rstrip("/")
        self._timeout = timeout

    #: 可用性 = 是否配置了 base_url（纯字符串判定），瞬时可得。
    local_probe = True

    def source_type(self) -> str:
        return SOURCE_TYPE_OTHER

    def is_available(self) -> bool:
        return bool(self._base_url)

    def describe(self) -> str:
        return f"通用HTTP行情接口: {self._base_url}"

    def _http_get(self, url: str) -> dict[str, Any]:
        import urllib.request

        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                body = resp.read()
        except Exception as exc:
            raise DataError(f"网络请求失败: {url}（{exc}）") from exc
        try:
            return json.loads(body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise DataError(f"网络返回非 JSON: {url}（{exc}）") from exc

    @staticmethod
    def _parse_json(raw: dict[str, Any]) -> pd.DataFrame:
        candles = raw.get("candles") or raw.get("data") or []
        if not candles:
            return pd.DataFrame()
        return pd.DataFrame(candles)

    def _generic_fetch(
        self, symbol: str, timeframe: str, since_ts: int | None
    ) -> pd.DataFrame | None:
        import urllib.parse

        url = (
            self._base_url
            + f"/candles?symbol={urllib.parse.quote(symbol)}"
            + f"&timeframe={urllib.parse.quote(timeframe)}"
        )
        if since_ts is not None:
            url += f"&since={since_ts}"
        df = self._parse_json(self._http_get(url))
        if df is None or len(df) == 0:
            return pd.DataFrame()
        return self._validate_df(df, symbol=symbol, timeframe=timeframe)

    def fetch_full(self, symbol: str, timeframe: str) -> pd.DataFrame:
        df = self._generic_fetch(symbol, timeframe, since_ts=None)
        if df is None or len(df) == 0:
            raise DataError(f"通用HTTP接口返回空数据: {symbol} {timeframe}")
        return df

    def fetch_incremental(self, symbol: str, timeframe: str, since_ts: int) -> pd.DataFrame:
        df = self._generic_fetch(symbol, timeframe, since_ts=since_ts)
        if df is None or len(df) == 0:
            return pd.DataFrame()
        return df


def _build_default_sources() -> list[DataSource]:
    """构造默认数据源注册表：每个数据源按类型**独立**成项。

    * 神机本地库 → ``Shenji``
    * 通用 HTTP 接口（若配置） → ``其他``
    * 每个网络 fetcher → 各自独立来源（TradingView / OKX / Dukascopy / 其他）

    **不做可用性探测**：这里用 :func:`all_network_sources` 而非
    :func:`list_network_sources`——后者会逐个 ``is_available()``，Binance /
    Dukascopy 走 TCP 探测，主机不可达时各卡满 3s 超时（实测列来源 6s）。
    可用性改由 :meth:`DataAcquisition.list_sources` 走缓存 + 后台线程判定。
    """
    cfg = data_acquisition_config()
    sources: list[DataSource] = [ShenjiLocalSource()]
    if cfg.data_api_url:
        sources.append(GenericHttpSource(base_url=cfg.data_api_url, timeout=cfg.timeout))
    for fetcher in all_network_sources(
        dukascopy_user=cfg.dukascopy_user or None,
        dukascopy_password=cfg.dukascopy_password or None,
    ):
        sources.append(TypedNetworkSource(fetcher))
    return sources


# ── 可用性：进程内缓存 + 后台探测（**绝不阻塞请求线程**）────────────────────

#: 可用性探测结果的缓存有效期（秒）。
#: 依据：可用性本质是「某主机 / 依赖能否连通」，分钟级稳定；取 5 分钟既能让
#: 切换画像瞬时响应，又不会在网络环境变化后长期锁死旧结果。
_AVAILABILITY_TTL = 300.0

#: 进程内可用性缓存：``key -> (available, 探测时刻 monotonic)``。
_AVAILABILITY_CACHE: dict[str, tuple[bool, float]] = {}
#: 正在后台探测中的 key（避免并发请求对同一源重复起线程）。
_AVAILABILITY_INFLIGHT: set[str] = set()
_AVAILABILITY_LOCK = threading.Lock()
#: 后台探测得到的真实描述（未探测时用占位，避免请求线程调 describe() 触发探测）。
_DESCRIPTION_CACHE: dict[str, str] = {}


def _availability_key(src: DataSource) -> str:
    """可用性缓存 key：**按数据源类型 + 包装类名（+ 内层 fetcher）**，不按实例身份。

    可用性的本质是「某一类数据源能不能连」，与具体实例无关；按实例（id()）
    做 key 会让每次新建对象都重新探测，缓存形同虚设。

    ⚠ 必须带上内层 fetcher 的类名：``TypedNetworkSource`` 把多个 fetcher 包装
    成同一个类，而 :data:`_FETCHER_TYPE_MAP` 对未登记的实现一律回落到
    ``其他``——只按「类型 + 包装类名」做 key 会让两个互不相关的 fetcher 共用
    一份可用性缓存（互相污染）。当前仅 ``TqsdkFetcher`` 落在这个回落分支上，
    但新增任何未登记的 fetcher 都会踩到，故在此一并消除隐患。
    """
    key = f"{src.source_type()}|{type(src).__name__}"
    inner = getattr(src, "_fetcher", None)
    if inner is not None:
        key = f"{key}|{type(inner).__name__}"
    return key


def _availability_from_cache(src: DataSource) -> bool | None:
    """取缓存中的可用性；**未探测或已过期返回 ``None``**（不是 False）。"""
    key = _availability_key(src)
    with _AVAILABILITY_LOCK:
        hit = _AVAILABILITY_CACHE.get(key)
    if hit is None:
        return None
    available, ts = hit
    if (time.monotonic() - ts) > _AVAILABILITY_TTL:
        return None
    return available


def _probe_sync(src: DataSource) -> bool:
    """**同步**探测并写缓存——只允许用于 :attr:`DataSource.local_probe` 的本地来源。

    网络来源禁止走这里（会把请求线程卡在不可达主机上），请用
    :func:`_probe_in_background`。
    """
    key = _availability_key(src)
    try:
        available = bool(src.is_available())
    except Exception:  # noqa: BLE001 - 探测异常一律视为不可用
        available = False
    try:
        description = src.describe()
    except Exception:  # noqa: BLE001 - 描述取不到用类型名兜底
        description = src.source_type()
    with _AVAILABILITY_LOCK:
        _AVAILABILITY_CACHE[key] = (available, time.monotonic())
        _DESCRIPTION_CACHE[key] = description
    return available


def _probe_one(src: DataSource, key: str) -> None:
    """后台线程：探测单个源（含描述），把结果写入缓存。

    ``describe()`` 也在这里一并取：Dukascopy / Binance 的 ``describe()`` 内部
    会调 ``is_available()``，放在后台取可避免请求线程被拖住。

    ``key`` 由调用方（请求线程）算好传入，这样本函数即使出现任何意外，
    ``finally`` 也能把 inflight 标记摘掉——否则该源会永远停在「检测中」。
    """
    try:
        try:
            available = bool(src.is_available())
        except Exception:  # noqa: BLE001 - 探测异常一律视为不可用，不能让后台线程炸掉
            available = False
        try:
            description = src.describe()
        except Exception:  # noqa: BLE001 - 描述取不到不影响可用性判定
            description = src.source_type()
    except Exception:  # noqa: BLE001 - 极端兜底：保证 finally 里变量已绑定
        available = False
        description = key
    finally:
        with _AVAILABILITY_LOCK:
            _AVAILABILITY_CACHE[key] = (available, time.monotonic())
            _DESCRIPTION_CACHE[key] = description
            _AVAILABILITY_INFLIGHT.discard(key)


def _description_for(src: DataSource) -> str:
    """取来源描述；未探测时返回 ``"（检测中…）"`` 占位，**不触发探测**。"""
    key = _availability_key(src)
    with _AVAILABILITY_LOCK:
        return _DESCRIPTION_CACHE.get(key, f"{src.source_type()}（检测中…）")


def _probe_in_background(sources: list[DataSource]) -> None:
    """对「未探测 / 已过期」的源**并行**发起后台探测，不阻塞当前请求。

    只探测传入的源——即已按市场画像 ``allowed_types`` 过滤过的，不再出现
    「先探测全部、再过滤画像」导致的无谓等待。用 daemon 线程：进程退出时
    不会被未完成的探测挂住。
    """
    pending: list[tuple[DataSource, str]] = []
    now = time.monotonic()
    with _AVAILABILITY_LOCK:
        for src in sources:
            key = _availability_key(src)
            hit = _AVAILABILITY_CACHE.get(key)
            if hit is not None and (now - hit[1]) <= _AVAILABILITY_TTL:
                continue
            if key in _AVAILABILITY_INFLIGHT:
                continue
            _AVAILABILITY_INFLIGHT.add(key)
            pending.append((src, key))
    for src, key in pending:
        threading.Thread(
            target=_probe_one,
            args=(src, key),
            daemon=True,
            name=f"probe-{key}",
        ).start()


def _cache_dirs_from_config() -> tuple[Path, ...]:
    """解析缓存目录（环境变量优先，否则用默认）；与数据**来源能力无关**。

    抽成模块级函数是为了让「只扫目录」的入口（``/api/acquisition/cached``）
    不必构造 :class:`DataAcquisition`——构造会 build 全部数据源并对网络 fetcher
    逐个做 TCP 探测，本环境 ``freeserv.dukascopy.com`` / ``api.binance.com``
    不可达，各卡满 3s 超时 → 首次调用 6s。列缓存文件只扫目录，本不需要这些。
    """
    custom = data_acquisition_config().cache_dir
    if custom:
        p = Path(custom)
        p.mkdir(parents=True, exist_ok=True)
        return (p,)
    return _DEFAULT_CACHE_DIRS


def _scan_cache_dirs(cache_dirs: tuple[Path, ...]) -> list[dict[str, Any]]:
    """扫描缓存目录列出 parquet / csv——**只做 iterdir + stat，无任何网络行为**。

    Args:
        cache_dirs: 待扫描目录（顺序即结果顺序）。

    Returns:
        每项含 ``name`` / ``path`` / ``size_mb`` / ``mtime`` / ``source`` 五个字段
        （前端与 ``webui/server.py`` 依赖，勿改名）；按文件名跨目录去重，
        先出现的目录优先。
    """
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for directory in cache_dirs:
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            if path.suffix.lower() not in (".parquet", ".csv"):
                continue
            key = path.name
            if key in seen:
                continue
            seen.add(key)
            stat = path.stat()
            out.append({
                "name": path.name,
                "path": str(path),
                "size_mb": round(stat.st_size / 1e6, 2),
                "mtime": time.strftime(
                    "%Y-%m-%d %H:%M", time.localtime(stat.st_mtime)
                ),
                "source": "cache",
            })
    return out


class DataAcquisition:
    """统一数据获取管理器。

    按优先级尝试已配置的来源（神机本地库优先 > 网络下载），
    自动完成增量 / 全量决策，结果落盘为 parquet 缓存。

    Usage::

        acq = DataAcquisition()
        path = acq.fetch("XAUUSD", "H1")  # → 本地 parquet 路径
        panel = miaosuan.data.loader.load(path)
    """

    def __init__(self, sources: list[DataSource] | None = None) -> None:
        if sources is None:
            sources = _build_default_sources()
        self._sources: list[DataSource] = sources
        self._cache_dirs: tuple[Path, ...] = self._resolve_cache_dirs()

    @staticmethod
    def _resolve_cache_dirs() -> tuple[Path, ...]:
        """确定缓存目录（环境变量优先，否则用默认）。

        委托模块级 :func:`_cache_dirs_from_config`（单一实现），保留本方法
        以兼容既有调用方。
        """
        return _cache_dirs_from_config()

    def _cache_path(self, symbol: str, timeframe: str, note: str | None = None) -> Path:
        """计算缓存文件路径（{symbol}_{timeframe}.parquet；备注非空时追加为 {symbol}_{timeframe}_{note}.parquet）。"""
        filename = f"{symbol}_{timeframe}.parquet"
        if note:
            slug = _slugify(note)
            if slug:
                filename = f"{symbol}_{timeframe}_{slug}.parquet"
        for directory in self._cache_dirs:
            directory.mkdir(parents=True, exist_ok=True)
            candidate = directory / filename
            if not candidate.exists() or candidate.is_file():
                return candidate
        # 所有目录都可用时，用第一个
        return self._cache_dirs[0] / filename

    def _read_cache_max_time(self, cache_path: Path) -> int | None:
        """读取本地缓存中 ``max(time)``（Unix 秒），无缓存返回 ``None``。"""
        if not cache_path.is_file():
            return None
        try:
            df = pd.read_parquet(cache_path)
        except Exception:
            return None
        if len(df) == 0 or "time" not in df.columns:
            return None
        return int(np.asarray(df["time"]).max())

    @staticmethod
    def _merge_and_dedup(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
        """合并增量与缓存，按 time 排序、去重（保留最后一条）。"""
        combined = pd.concat([old, new], ignore_index=True)
        combined = combined.sort_values("time")
        combined = combined[~combined["time"].duplicated(keep="last")]
        return combined.reset_index(drop=True)

    @staticmethod
    def _write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
        """原子写入 parquet（写 .tmp 后 os.replace，防多进程损坏）。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.parent / (path.name + ".tmp")
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)

    def fetch(
        self,
        symbol: str,
        timeframe: str,
        since: int | None = None,
        source: str | None = None,
        note: str | None = None,
        allowed_types: list[str] | None = None,
    ) -> Path:
        """获取行情数据并缓存为本地 parquet。

        Args:
            symbol: 品种标识（如 ``"XAUUSD"``）。
            timeframe: 周期标识（如 ``"H1"``）。
            since: 可选的增量起始 Unix 秒（一般不需要传，模块自动判定）。
            source: 指定单一数据来源**类型**（如 ``"Shenji"`` / ``"TradingView"`` / ``"OKX"``
                / ``"Dukascopy"`` / ``"其他"``）；空 = 按优先级自动。
            note: 备注，非空时追加到缓存文件名（``{symbol}_{timeframe}_{note}.parquet``）。
            allowed_types: 仅允许的来源类型列表（按市场画像约束）；空 = 不限制。

        Returns:
            本地 parquet 文件路径，可直接传给 ``loader.load()``。

        Raises:
            DataError: 无可用来源、来源返回空数据或格式错误时。
        """
        cache_path = self._cache_path(symbol, timeframe, note=note)
        max_time = self._read_cache_max_time(cache_path)
        sources = self._select_sources(source, allowed_types=allowed_types)

        # 确定增量起点：显式 since > 缓存 max_time > None（全量）
        incremental_since = since
        if incremental_since is None and max_time is not None:
            incremental_since = max_time

        # 尝试增量获取
        if incremental_since is not None:
            new_df = self._try_fetch_incremental(sources, symbol, timeframe, incremental_since)
            if new_df is not None and len(new_df) > 0:
                old_df = pd.DataFrame()
                if cache_path.is_file():
                    try:
                        old_df = pd.read_parquet(cache_path)
                    except Exception:
                        old_df = pd.DataFrame()
                merged = self._merge_and_dedup(old_df, new_df) if len(old_df) > 0 else new_df
                self._write_parquet_atomic(merged, cache_path)
                return cache_path
            # 增量无新数据 → 直接用已有缓存
            if cache_path.is_file():
                return cache_path

        # 首跑或增量失败 → 全量获取
        full_df = self._try_fetch_full(sources, symbol, timeframe)
        if full_df is not None and len(full_df) > 0:
            self._write_parquet_atomic(full_df, cache_path)
            return cache_path

        # 全量也失败 → 尝试用已有缓存（可能过期但总比没有好）
        if cache_path.is_file():
            return cache_path

        available = " / ".join(s.describe() for s in sources)
        raise DataError(
            f"无法获取 {symbol} {timeframe} 的行情数据: 无可用来源或来源均失败",
            context={"symbol": symbol, "timeframe": timeframe, "source": source, "sources": available},
        )

    def _select_sources(
        self, source_name: str | None, allowed_types: list[str] | None = None
    ) -> list[DataSource]:
        """按「数据源类型」筛选来源；空名返回全部（受 ``allowed_types`` 约束）。

        Args:
            source_name: 指定的单一数据源类型（如 ``"Shenji"`` / ``"TradingView"``）；
                空 = 按 ``allowed_types`` 内全部来源依次尝试。
            allowed_types: 仅允许的来源类型列表（来自市场画像）；空 = 不限制。
        """
        if allowed_types is not None:
            pool = [s for s in self._sources if s.source_type() in allowed_types]
        else:
            pool = list(self._sources)
        if not source_name:
            return pool
        chosen = [s for s in pool if s.source_type() == source_name]
        if not chosen:
            avail = sorted({s.source_type() for s in pool})
            raise DataError(
                f"未找到指定的数据来源：{source_name}",
                context={"available": avail},
            )
        return chosen

    def _try_fetch_full(
        self, sources: list[DataSource], symbol: str, timeframe: str
    ) -> pd.DataFrame | None:
        """按给定来源列表尝试全量获取。"""
        for src in sources:
            if not src.is_available():
                continue
            try:
                return src.fetch_full(symbol, timeframe)
            except Exception:
                continue
        return None

    def _try_fetch_incremental(
        self, sources: list[DataSource], symbol: str, timeframe: str, since_ts: int
    ) -> pd.DataFrame | None:
        """按给定来源列表尝试增量获取。"""
        for src in sources:
            if not src.is_available():
                continue
            try:
                df = src.fetch_incremental(symbol, timeframe, since_ts)
                if df is not None and len(df) > 0:
                    return df
            except Exception:
                continue
        return None

    def list_cached(self) -> list[dict[str, Any]]:
        """列出本地缓存中已有的数据文件（供 UI 展示）。

        委托 :func:`_scan_cache_dirs`（只扫目录，无网络行为）；本方法保留以
        兼容既有调用方与测试。
        """
        return _scan_cache_dirs(self._cache_dirs)

    def list_sources(self, allowed_types: list[str] | None = None) -> list[dict[str, Any]]:
        """列出已配置的数据来源及其可用性（供 UI 展示）。

        可用性**绝不阻塞请求**：命中进程内缓存直接返回已知值；未探测 / 已过期
        则返回 ``None``（前端显示「检测中」）并**后台并行探测**，结果写入缓存
        供下一次请求使用。因此切换市场画像是瞬时的，即使有主机不可达。

        乐观展示会不会害用户点到不可用的源？不会——:meth:`fetch` 会收集每个
        来源的 ``failures`` 并透传到错误消息（见 ``NetworkSource.fetch_full`` /
        ``_try_fetch_*``），用户点下去拿到的是明确的失败原因，而非静默错误数据。

        Args:
            allowed_types: 仅返回这些「数据源类型」的来源（按市场画像过滤用）。
                先按本参数过滤、再只探测留下的源——不做无谓的探测。
        """
        out: list[dict[str, Any]] = []
        selected: list[DataSource] = []
        for src in self._sources:
            st = src.source_type()
            if allowed_types is not None and st not in allowed_types:
                continue
            selected.append(src)
            known = _availability_from_cache(src)
            if known is None and getattr(src, "local_probe", False):
                # 本地来源（文件 / 配置判定）瞬时可得 → 同步判定，当场给真实值
                known = _probe_sync(src)
            out.append({
                "name": type(src).__name__,
                "source_type": st,
                "description": _description_for(src),
                "available": known,  # None = 尚未探测（前端显示「检测中」）
            })
        _probe_in_background(selected)
        return out


#: 模块级单例（懒初始化，首次调用时构造）。
_acquisition: DataAcquisition | None = None


def _get_acquisition() -> DataAcquisition:
    global _acquisition
    if _acquisition is None:
        _acquisition = DataAcquisition()
    return _acquisition


def fetch(
    symbol: str,
    timeframe: str,
    since: int | None = None,
    source: str | None = None,
    note: str | None = None,
    allowed_types: list[str] | None = None,
) -> Path:
    """模块级快捷入口（等价于 ``DataAcquisition().fetch(...)``）。"""
    return _get_acquisition().fetch(
        symbol, timeframe, since=since, source=source, note=note, allowed_types=allowed_types
    )


def list_cached() -> list[dict[str, Any]]:
    """模块级快捷入口：列出本地缓存。

    刻意**不**走 :func:`_get_acquisition()`：列缓存文件只是扫目录，构造
    ``DataAcquisition`` 却会 build 全部数据源并逐个做网络 TCP 探测（不可达
    主机各卡满超时 → 首次调用 6s）。走 :func:`_scan_cache_dirs` 只做 IO。
    """
    return _scan_cache_dirs(_cache_dirs_from_config())


def list_sources(allowed_types: list[str] | None = None) -> list[dict[str, Any]]:
    """模块级快捷入口：列出已配置来源（可按 ``allowed_types`` 过滤）。"""
    return _get_acquisition().list_sources(allowed_types=allowed_types)
