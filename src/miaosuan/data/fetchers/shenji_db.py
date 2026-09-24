# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for full license text.

"""妙算本地 SQLite 数据库读取器（只读）。

读取妙算落库的 ``ohlcv`` 表（列：``timeframe, timestamp, open, high, low,
close, volume``）。**该库不存品种列**——它只保存单一品种（默认 XAUUSD），
因此 :meth:`fetch_full` / :meth:`fetch_incremental` 的 ``symbol`` 参数被
**显式忽略**并在文档中说明，调用方无需、也无法按品种区分。

**只读访问**：使用 ``sqlite3`` 的 ``mode=ro`` URI 打开，绝不写入（架构 §3.2 硬约束）。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from ...errors import DataError
from .base import BaseFetcher

__all__ = ["ShenjiDBFetcher"]


class ShenjiDBFetcher(BaseFetcher):
    """妙算平台本地 SQLite 行情库读取器（只读）。"""

    source_name = "妙算_SQLite"

    def __init__(self, db_path: str | None) -> None:
        """初始化。

        Args:
            db_path: 妙算 SQLite 数据库文件路径；为 ``None`` 或不存在时视为不可用。
        """
        self._db_path: str | None = db_path

    def is_available(self) -> bool:
        return self._db_path is not None and Path(self._db_path).is_file()

    def describe(self) -> str:
        if self._db_path:
            return f"ShenjiDB: {self._db_path}"
        return "ShenjiDB: (未配置)"

    # ── 内部 ─────────────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        if not self.is_available():
            raise DataError(
                f"妙算本地数据库不可用: {self._db_path!r}",
                context={"db_path": self._db_path},
            )
        # 只读 URI 打开，防止任何写操作（架构 §3.2 硬约束）
        uri = "file:" + self._db_path + "?mode=ro"
        return sqlite3.connect(uri, uri=True)

    def _query(self, timeframe: str, since_ts: int | None) -> pd.DataFrame:
        query = (
            "SELECT timestamp AS time, open, high, low, close, volume "
            "FROM ohlcv WHERE timeframe = ?"
        )
        params: list = [timeframe]
        if since_ts is not None:
            query += " AND timestamp > ?"
            params.append(int(since_ts))
        query += " ORDER BY timestamp ASC"
        conn = self._connect()
        try:
            df = pd.read_sql_query(query, conn, params=params)
        finally:
            conn.close()
        return self._finalize(df)

    def fetch_full(self, symbol: str, timeframe: str) -> pd.DataFrame:
        # 注意：妙算库不含 symbol 列，symbol 被有意忽略（库仅存单一品种）。
        _ = symbol
        return self._query(timeframe, None)

    def fetch_incremental(
        self, symbol: str, timeframe: str, since_ts: int
    ) -> pd.DataFrame:
        _ = symbol
        return self._query(timeframe, since_ts)
