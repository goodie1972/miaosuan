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
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..errors import DataError

__all__ = [
    "DataSource",
    "ShenjiLocalSource",
    "NetworkSource",
    "DataAcquisition",
    "fetch",
]

#: 默认本地缓存目录列表（与 ``server.py`` 的 ``_DATA_DIRS`` 对齐）。
_DEFAULT_CACHE_DIRS: tuple[Path, ...] = (
    Path(r"D:\K线数据"),
    Path(__file__).resolve().parents[2] / "data",
)

#: 环境变量名：神机本地数据库路径。
_ENV_SHENJI_DB: str = "MIAOSUAN_SHENJI_DB_PATH"

#: 环境变量名：网络数据 API 地址。
_ENV_DATA_API_URL: str = "MIAOSUAN_DATA_API_URL"

#: 环境变量名：自定义缓存目录（覆盖默认）。
_ENV_CACHE_DIR: str = "MIAOSUAN_DATA_CACHE_DIR"

#: 环境变量名：网络请求超时秒数。
_ENV_TIMEOUT: str = "MIAOSUAN_DATA_TIMEOUT"

#: 默认网络超时（秒）。
_DEFAULT_TIMEOUT: int = 30

#: 支持的 OHLCV 列名（与 ``loader.panel_from_frame`` 对齐）。
_REQUIRED_COLUMNS = ("time", "open", "high", "low", "close", "volume", "tick_volume")


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
    """神机本地数据库数据源。

    通过神机提供的只读查询接口拉取已落库的行情数据。
    当前为**桩实现**：当环境变量 ``MIAOSUAN_SHENJI_DB_PATH`` 指向一个
    SQLite 数据库文件时，用标准库 ``sqlite3`` 执行查询；否则视为不可用。
    """

    def __init__(self) -> None:
        self._db_path: str | None = os.environ.get(_ENV_SHENJI_DB)

    def is_available(self) -> bool:
        return self._db_path is not None and Path(self._db_path).is_file()

    def describe(self) -> str:
        if self._db_path:
            return f"Shenji DB: {self._db_path}"
        return "Shenji DB: (未配置)"

    def fetch_full(self, symbol: str, timeframe: str) -> pd.DataFrame:
        if not self.is_available():
            raise DataError(f"神机本地数据库不可用: {self._db_path!r}")
        import sqlite3

        conn = sqlite3.connect(self._db_path)
        try:
            query = (
                "SELECT time, open, high, low, close, tick_volume "
                "FROM candles WHERE symbol = ? AND timeframe = ? ORDER BY time"
            )
            df = pd.read_sql_query(query, conn, params=(symbol, timeframe))
        finally:
            conn.close()
        return self._validate_df(df, symbol=symbol, timeframe=timeframe)

    def fetch_incremental(
        self, symbol: str, timeframe: str, since_ts: int
    ) -> pd.DataFrame:
        if not self.is_available():
            raise DataError(f"神机本地数据库不可用: {self._db_path!r}")
        import sqlite3

        conn = sqlite3.connect(self._db_path)
        try:
            query = (
                "SELECT time, open, high, low, close, tick_volume "
                "FROM candles WHERE symbol = ? AND timeframe = ? AND time > ? ORDER BY time"
            )
            df = pd.read_sql_query(query, conn, params=(symbol, timeframe, since_ts))
        finally:
            conn.close()
        if len(df) == 0:
            return pd.DataFrame()
        return self._validate_df(df, symbol=symbol, timeframe=timeframe)


class NetworkSource(DataSource):
    """网络下载数据源。

    通过 HTTP 从数据服务商 / 公开源按需下载行情数据。
    当前为**桩实现**：当环境变量 ``MIAOSUAN_DATA_API_URL`` 设置了基础 URL 时，
    用 ``urllib`` 发送请求；否则视为不可用。
    """

    def __init__(self) -> None:
        self._base_url: str | None = os.environ.get(_ENV_DATA_API_URL)
        raw_timeout = os.environ.get(_ENV_TIMEOUT, "")
        try:
            self._timeout: int = int(raw_timeout) if raw_timeout else _DEFAULT_TIMEOUT
        except ValueError:
            self._timeout = _DEFAULT_TIMEOUT

    def is_available(self) -> bool:
        return self._base_url is not None and len(self._base_url) > 0

    def describe(self) -> str:
        if self._base_url:
            return f"Network API: {self._base_url}"
        return "Network API: (未配置)"

    def fetch_full(self, symbol: str, timeframe: str) -> pd.DataFrame:
        if not self.is_available():
            raise DataError(f"网络数据 API 不可用: {self._base_url!r}")
        import urllib.parse
        import urllib.request

        url = (
            self._base_url.rstrip("/")
            + f"/candles?symbol={urllib.parse.quote(symbol)}"
            + f"&timeframe={urllib.parse.quote(timeframe)}"
        )
        raw = self._http_get(url)
        df = self._parse_json(raw)
        return self._validate_df(df, symbol=symbol, timeframe=timeframe)

    def fetch_incremental(
        self, symbol: str, timeframe: str, since_ts: int
    ) -> pd.DataFrame:
        if not self.is_available():
            raise DataError(f"网络数据 API 不可用: {self._base_url!r}")
        import urllib.parse
        import urllib.request

        url = (
            self._base_url.rstrip("/")
            + f"/candles?symbol={urllib.parse.quote(symbol)}"
            + f"&timeframe={urllib.parse.quote(timeframe)}"
            + f"&since={since_ts}"
        )
        raw = self._http_get(url)
        df = self._parse_json(raw)
        if len(df) == 0:
            return pd.DataFrame()
        return self._validate_df(df, symbol=symbol, timeframe=timeframe)

    def _http_get(self, url: str) -> dict[str, Any]:
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
            sources = [ShenjiLocalSource(), NetworkSource()]
        self._sources: list[DataSource] = sources
        self._cache_dirs: tuple[Path, ...] = self._resolve_cache_dirs()

    @staticmethod
    def _resolve_cache_dirs() -> tuple[Path, ...]:
        """确定缓存目录（环境变量优先，否则用默认）。"""
        custom = os.environ.get(_ENV_CACHE_DIR, "")
        if custom:
            p = Path(custom)
            p.mkdir(parents=True, exist_ok=True)
            return (p,)
        return _DEFAULT_CACHE_DIRS

    def _cache_path(self, symbol: str, timeframe: str) -> Path:
        """计算缓存文件路径（{symbol}_{timeframe}.parquet）。"""
        filename = f"{symbol}_{timeframe}.parquet"
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

    def fetch(self, symbol: str, timeframe: str, since: int | None = None) -> Path:
        """获取行情数据并缓存为本地 parquet。

        Args:
            symbol: 品种标识（如 ``"XAUUSD"``）。
            timeframe: 周期标识（如 ``"H1"``）。
            since: 可选的增量起始 Unix 秒（一般不需要传，模块自动判定）。

        Returns:
            本地 parquet 文件路径，可直接传给 ``loader.load()``。

        Raises:
            DataError: 无可用来源、来源返回空数据或格式错误时。
        """
        cache_path = self._cache_path(symbol, timeframe)
        max_time = self._read_cache_max_time(cache_path)

        # 确定增量起点：显式 since > 缓存 max_time > None（全量）
        incremental_since = since
        if incremental_since is None and max_time is not None:
            incremental_since = max_time

        # 尝试增量获取
        if incremental_since is not None:
            new_df = self._try_fetch_incremental(symbol, timeframe, incremental_since)
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
        full_df = self._try_fetch_full(symbol, timeframe)
        if full_df is not None and len(full_df) > 0:
            self._write_parquet_atomic(full_df, cache_path)
            return cache_path

        # 全量也失败 → 尝试用已有缓存（可能过期但总比没有好）
        if cache_path.is_file():
            return cache_path

        available = " / ".join(s.describe() for s in self._sources)
        raise DataError(
            f"无法获取 {symbol} {timeframe} 的行情数据: 无可用来源或来源均失败",
            context={"symbol": symbol, "timeframe": timeframe, "sources": available},
        )

    def _try_fetch_full(self, symbol: str, timeframe: str) -> pd.DataFrame | None:
        """按优先级尝试全量获取。"""
        for src in self._sources:
            if not src.is_available():
                continue
            try:
                return src.fetch_full(symbol, timeframe)
            except Exception:
                continue
        return None

    def _try_fetch_incremental(
        self, symbol: str, timeframe: str, since_ts: int
    ) -> pd.DataFrame | None:
        """按优先级尝试增量获取。"""
        for src in self._sources:
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

    def list_sources(self) -> list[dict[str, Any]]:
        """列出已配置的数据来源及其可用性（供 UI 展示）。"""
        return [
            {
                "name": type(src).__name__,
                "description": src.describe(),
                "available": src.is_available(),
            }
            for src in self._sources
        ]


#: 模块级单例（懒初始化，首次调用时构造）。
_acquisition: DataAcquisition | None = None


def _get_acquisition() -> DataAcquisition:
    global _acquisition
    if _acquisition is None:
        _acquisition = DataAcquisition()
    return _acquisition


def fetch(symbol: str, timeframe: str, since: int | None = None) -> Path:
    """模块级快捷入口（等价于 ``DataAcquisition().fetch(...)``）。"""
    return _get_acquisition().fetch(symbol, timeframe, since=since)


def list_cached() -> list[dict[str, Any]]:
    """模块级快捷入口：列出本地缓存。"""
    return _get_acquisition().list_cached()


def list_sources() -> list[dict[str, Any]]:
    """模块级快捷入口：列出已配置来源。"""
    return _get_acquisition().list_sources()
