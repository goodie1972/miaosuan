# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for full license text.

"""数据获取器抽象基类（与 :class:`~miaosuan.data.acquisition.DataSource` 平级的鸭子类型）。

本模块定义 :class:`BaseFetcher`，供 ``acquisition.py`` 的 ``DataSource`` 委托。
各具体实现（神机本地库、TradingView、akshare、tqsdk、OKX、Dukascopy）都继承它，
对外暴露统一接口：

* ``is_available()`` —— 当前环境是否可用（依赖 / 凭据就绪）；
* ``describe()`` —— 人类可读描述；
* ``fetch_full(symbol, timeframe)`` —— 全量历史；
* ``fetch_incremental(symbol, timeframe, since_ts)`` —— ``since_ts`` 之后的增量。

所有 fetcher 返回的 ``DataFrame`` 必须满足统一列契约
``time(int) / open / high / low / close(float) / volume(float)``，并可选地包含
``tick_volume``（= volume，兼容 loader）。行按 ``time`` 升序、无重复时间戳。

设计约束（架构 §9.7 / 依赖方向铁律）：

* **禁止**读取环境变量——所有外部配置（代理、凭据、路径）一律经构造参数注入；
* 第三方重依赖（``tvdatafeed`` / ``akshare`` / ``tqsdk``）一律**方法内惰性导入**
  并包裹 ``try/except ImportError``，缺依赖时 ``is_available()`` 优雅返回 ``False``，
  保证 ``acquisition.py`` 在缺依赖环境下仍可正常 import。
"""

from __future__ import annotations

import pandas as pd

from ...errors import DataError

__all__ = ["BaseFetcher"]

#: 标准输出列（顺序即最终 DataFrame 列顺序）。
_REQUIRED_COLUMNS = ("time", "open", "high", "low", "close", "volume")


class BaseFetcher:
    """数据获取器基类（鸭子类型，供 ``acquisition`` 委托）。"""

    #: 来源名称（用于 describe / 日志）。
    source_name: str = "base"

    def is_available(self) -> bool:
        """当前来源是否可用。子类应优雅处理依赖缺失。"""
        return False

    def describe(self) -> str:
        """人类可读的来源描述。"""
        return f"{self.source_name}: (unavailable)"

    def fetch_full(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """全量获取 ``symbol`` + ``timeframe`` 的历史数据。"""
        raise NotImplementedError(f"{type(self).__name__} 未实现 fetch_full")

    def fetch_incremental(
        self, symbol: str, timeframe: str, since_ts: int
    ) -> pd.DataFrame:
        """增量获取 ``since_ts``（Unix 秒）之后的数据。

        默认实现：拉全量后过滤（多数网络源不支持服务端 since）。具体子类
        若支持服务端增量可覆盖本方法。
        """
        full = self.fetch_full(symbol, timeframe)
        if full is None or len(full) == 0:
            return self._finalize(pd.DataFrame())
        return self._finalize(full[full["time"] > since_ts])

    # ── 共享工具 ───────────────────────────────────────────────────────────

    @staticmethod
    def _finalize(df: pd.DataFrame) -> pd.DataFrame:
        """把任意 OHLCV 形态规整为统一契约并排序 / 去重。

        Args:
            df: 至少含 ``time/open/high/low/close/volume`` 的 DataFrame。

        Returns:
            规整后的 DataFrame（列顺序 ``time, open, high, low, close, volume,
            tick_volume``；按 ``time`` 升序、按 ``time`` 去重保留末值）。
            空输入返回带标准列的空 DataFrame。
        """
        if df is None or len(df) == 0:
            return pd.DataFrame(columns=[*_REQUIRED_COLUMNS, "tick_volume"])

        df = df.copy()
        missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise DataError(
                f"fetcher 返回数据缺少必需列: {missing}",
                context={"missing": missing},
            )

        # 类型归一
        df["time"] = df["time"].astype("int64")
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = df[col].astype("float64")

        # tick_volume 兼容列（= volume，供 loader 使用）
        if "tick_volume" not in df.columns:
            df["tick_volume"] = df["volume"]

        # 排序 + 去重（保留末值，避免网络源返回重复时间戳）
        df = df.sort_values("time").reset_index(drop=True)
        df = df[~df["time"].duplicated(keep="last")].reset_index(drop=True)
        return df[[*_REQUIRED_COLUMNS, "tick_volume"]]
