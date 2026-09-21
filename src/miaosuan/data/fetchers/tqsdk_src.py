# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for full license text.

"""天勤（tqsdk）数据源（惰性导入 ``tqsdk``）。"""

from __future__ import annotations

import pandas as pd

from ...errors import DataError
from .base import BaseFetcher

__all__ = ["TqsdkFetcher"]

#: 妙算周期 -> tqsdk ``get_kline_serial`` 的 duration_seconds。
_TQ_SECONDS = {
    "M1": 60,
    "M5": 300,
    "M15": 900,
    "M30": 1800,
    "H1": 3600,
    "H4": 14400,
    "D1": 86400,
    "W1": 604800,
}
#: tqsdk 单次最大序列长度（文档限制）。
_TQ_MAX_BARS = 8000


class TqsdkFetcher(BaseFetcher):
    """天勤量化（tqsdk）K 线读取器。"""

    source_name = "tqsdk"

    def __init__(
        self,
        tq_symbol: str | None = None,
        user: str | None = None,
        password: str | None = None,
        max_bars: int = _TQ_MAX_BARS,
    ) -> None:
        """初始化。

        Args:
            tq_symbol: tqsdk 合约代码（默认 ``KQ.i@XAUUSD`` 连续合约占位）；
                实盘应覆盖为账户实际可交易的品种。
            user: 快期账户（可选）。提供则与 ``password`` 以 ``TqAuth`` 登录。
            password: 快期密码（可选）。
            max_bars: 全量拉取的最大根数。
        """
        self._symbol = tq_symbol or "KQ.i@XAUUSD"
        self._user = user
        self._password = password
        self._max_bars = max_bars

    def is_available(self) -> bool:
        try:
            import tqsdk  # noqa: F401
        except Exception:
            return False
        return True

    def describe(self) -> str:
        if self.is_available():
            return f"tqsdk: {self._symbol}"
        return "tqsdk: (未安装 tqsdk)"

    @staticmethod
    def _to_unix_seconds(series: pd.Series) -> pd.Series:
        """tqsdk 的 ``datetime`` 列（tz-aware）转为 Unix 秒（int）。"""
        dt = pd.to_datetime(series, errors="coerce", utc=True)
        dt = dt.dt.tz_localize(None)
        return (dt.astype("int64") // 1_000_000_000).astype("int64")

    def fetch_full(self, symbol: str, timeframe: str) -> pd.DataFrame:
        if not self.is_available():
            raise DataError("tqsdk 不可用：未安装 tqsdk")
        seconds = _TQ_SECONDS.get(timeframe.upper())
        if seconds is None:
            raise DataError(f"tqsdk 不支持周期: {timeframe}")
        try:
            from tqsdk import TqApi, TqAuth, TqSim
        except ImportError as exc:
            raise DataError("未安装 tqsdk") from exc

        try:
            if self._user and self._password:
                api = TqApi(auth=TqAuth(self._user, self._password))
            else:
                api = TqApi(TqSim())
        except Exception as exc:
            raise DataError(f"tqsdk 连接失败: {exc}") from exc

        try:
            df = api.get_kline_serial(
                self._symbol, seconds, data_length=self._max_bars
            )
            if df is None or len(df) == 0:
                raise DataError(f"tqsdk 无数据：{self._symbol}")
            api.wait_update()
        finally:
            try:
                api.close()
            except Exception:
                pass

        out = pd.DataFrame(
            {
                "time": self._to_unix_seconds(df["datetime"]),
                "open": df["open"].astype("float64"),
                "high": df["high"].astype("float64"),
                "low": df["low"].astype("float64"),
                "close": df["close"].astype("float64"),
                "volume": df["volume"].astype("float64"),
            }
        )
        return self._finalize(out)
