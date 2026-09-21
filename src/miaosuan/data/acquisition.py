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
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import data_acquisition_config
from ..data.fetchers import BaseFetcher, ShenjiDBFetcher, list_network_sources
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
        # 1) 通用 HTTP JSON 接口
        if self._base_url:
            try:
                df = self._generic_fetch(symbol, timeframe, since_ts=None)
                if df is not None and len(df) > 0:
                    return self._validate_df(df, symbol=symbol, timeframe=timeframe)
            except Exception:
                pass
        # 2) 网络 fetcher 依次尝试
        for src in self._sources:
            if not src.is_available():
                continue
            try:
                df = src.fetch_full(symbol, timeframe)
                if df is not None and len(df) > 0:
                    return self._validate_df(df, symbol=symbol, timeframe=timeframe)
            except Exception:
                continue
        raise DataError(
            f"无可用网络来源: {symbol} {timeframe}",
            context={"sources": [s.describe() for s in self._sources]},
        )

    def fetch_incremental(
        self, symbol: str, timeframe: str, since_ts: int
    ) -> pd.DataFrame:
        # 通用 HTTP 接口支持 since 参数；网络 fetcher 内部按 since 过滤
        if self._base_url:
            try:
                df = self._generic_fetch(symbol, timeframe, since_ts=since_ts)
                if df is not None and len(df) > 0:
                    return self._validate_df(df, symbol=symbol, timeframe=timeframe)
            except Exception:
                pass
        for src in self._sources:
            if not src.is_available():
                continue
            try:
                df = src.fetch_incremental(symbol, timeframe, since_ts)
                if df is not None and len(df) > 0:
                    return self._validate_df(df, symbol=symbol, timeframe=timeframe)
            except Exception:
                continue
        raise DataError(
            f"无可用网络来源（增量）: {symbol} {timeframe}",
            context={"sources": [s.describe() for s in self._sources]},
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
    * 每个可用网络 fetcher → 各自独立来源（TradingView / OKX / Dukascopy / 其他）
    """
    cfg = data_acquisition_config()
    sources: list[DataSource] = [ShenjiLocalSource()]
    if cfg.data_api_url:
        sources.append(GenericHttpSource(base_url=cfg.data_api_url, timeout=cfg.timeout))
    for fetcher in list_network_sources(
        dukascopy_user=cfg.dukascopy_user or None,
        dukascopy_password=cfg.dukascopy_password or None,
    ):
        sources.append(TypedNetworkSource(fetcher))
    return sources


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
        """确定缓存目录（环境变量优先，否则用默认）。"""
        custom = data_acquisition_config().cache_dir
        if custom:
            p = Path(custom)
            p.mkdir(parents=True, exist_ok=True)
            return (p,)
        return _DEFAULT_CACHE_DIRS

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
        """列出本地缓存中已有的数据文件（供 UI 展示）。"""
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for directory in self._cache_dirs:
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

    def list_sources(self, allowed_types: list[str] | None = None) -> list[dict[str, Any]]:
        """列出已配置的数据来源及其可用性（供 UI 展示）。

        Args:
            allowed_types: 仅返回这些「数据源类型」的来源（按市场画像过滤用）。
        """
        out: list[dict[str, Any]] = []
        for src in self._sources:
            st = src.source_type()
            if allowed_types is not None and st not in allowed_types:
                continue
            out.append({
                "name": type(src).__name__,
                "source_type": st,
                "description": src.describe(),
                "available": src.is_available(),
            })
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
    """模块级快捷入口：列出本地缓存。"""
    return _get_acquisition().list_cached()


def list_sources(allowed_types: list[str] | None = None) -> list[dict[str, Any]]:
    """模块级快捷入口：列出已配置来源（可按 ``allowed_types`` 过滤）。"""
    return _get_acquisition().list_sources(allowed_types=allowed_types)
