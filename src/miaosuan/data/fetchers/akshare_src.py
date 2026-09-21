# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for full license text.

"""AKShare 数据源（A 股 / 国内商品期货历史 K 线，惰性导入 ``akshare``）。"""

from __future__ import annotations

import datetime as _dt

import pandas as pd

from ...errors import DataError
from .base import BaseFetcher

__all__ = ["AkshareFetcher"]

#: 妙算周期 -> akshare 期货分钟周期字符串（``futures_zh_minute_sina`` 的 period 取值）。
_AK_MINUTE = {"M1": "1", "M5": "5", "M15": "15", "M30": "30", "H1": "60"}
#: 期货日线及以上周期走 ``futures_zh_daily_sina``。
_AK_DAILY = {"H4", "D1", "W1"}

#: 妙算周期 -> A 股日/周线 ``stock_zh_a_hist`` 的 period 取值。
_AK_EQUITY_PERIOD = {"D1": "daily", "W1": "weekly"}
#: 妙算周期 -> A 股分钟线 ``stock_zh_a_hist_min_em`` 的 period 取值。
_AK_EQUITY_MINUTE = {"M1": "1", "M5": "5", "M15": "15", "M30": "30", "H1": "60"}

#: 默认期货品种（沪金连续）。
_DEFAULT_AK_SYMBOL = "au0"
#: 默认 A 股品种（贵州茅台，占位）。
_DEFAULT_AK_EQUITY_SYMBOL = "600519"

#: A 股历史起始日（``stock_zh_a_hist`` 需要 start_date / end_date）。
_AK_EQUITY_START = "19900101"


class AkshareFetcher(BaseFetcher):
    """AKShare 行情读取器（A 股 / 国内商品期货）。

    市场类型由构造参数 ``market`` 选择：

    * ``"futures"``（默认）：走 ``futures_zh_daily_sina`` / ``futures_zh_minute_sina``，
      品种代码形如 ``au0``（沪金连续）、``MA0``（甲醇主力）、``UR0``（尿素主力）。
      AKShare 免费接口**不直接暴露 XAUUSD 现货**，故外汇黄金默认映射沪金连续。
    * ``"equity"``：走 ``stock_zh_a_hist``（日线 / 周线）与 ``stock_zh_a_hist_min_em``
      （分钟线），品种代码为 6 位 A 股代码（如 ``600519``）。

    :meth:`fetch_full` 的 ``symbol`` 参数**优先于**构造期默认品种：调用方（如 Web UI
    按市场画像传入的品种）给了非空 ``symbol`` 就按它拉取，否则回退 ``ak_symbol``。
    列名同时兼容中文（日期/开盘/最高/最低/收盘/成交量）与英文，统一转换为标准契约。

    若 ``akshare`` 未安装，``is_available()`` 优雅返回 ``False``。
    """

    source_name = "akshare"

    def __init__(self, ak_symbol: str | None = None, market: str = "futures") -> None:
        """初始化。

        Args:
            ak_symbol: 默认 akshare 品种代码（``futures`` 默认 ``au0``；``equity`` 默认 ``600519``）。
            market: ``"futures"``（国内商品 / 金融期货）或 ``"equity"``（A 股）。
        """
        if market not in ("futures", "equity"):
            raise DataError(
                f"akshare 不支持市场类型: {market}（可选 futures / equity）",
                context={"market": market},
            )
        self._market = market
        self._symbol = ak_symbol or (
            _DEFAULT_AK_SYMBOL if market == "futures" else _DEFAULT_AK_EQUITY_SYMBOL
        )

    def is_available(self) -> bool:
        # 用 find_spec 而非真正 import：akshare 是巨型包，首次导入耗时数秒，
        # 会显著拖慢「数据源列表」接口；真正的导入延迟到 fetch_full 内执行。
        try:
            import importlib.util

            return importlib.util.find_spec("akshare") is not None
        except Exception:
            return False

    def describe(self) -> str:
        if self.is_available():
            return f"akshare[{self._market}]: {self._symbol}"
        return "akshare: (未安装 akshare)"

    @staticmethod
    def _to_unix_seconds(series: pd.Series) -> pd.Series:
        """把日期列（字符串 / datetime，可能 tz-aware）转为 Unix 秒（int）。"""
        dt = pd.to_datetime(series, errors="coerce")
        if getattr(dt.dt, "tz", None) is not None:
            dt = dt.dt.tz_convert("UTC").dt.tz_localize(None)
        return (dt.astype("int64") // 1_000_000_000).astype("int64")

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """把 akshare 返回的中 / 英列名规整为 time/open/high/low/close/volume。"""
        aliases = {
            "time": ["time", "日期", "date", "datetime", "时间"],
            "open": ["open", "开盘"],
            "high": ["high", "最高"],
            "low": ["low", "最低"],
            "close": ["close", "收盘"],
            "volume": ["volume", "成交量"],
        }
        rename: dict[str, str] = {}
        lower_cols = {c.lower(): c for c in df.columns}
        for std, candidates in aliases.items():
            for cand in candidates:
                key = cand.lower()
                if key in lower_cols:
                    rename[lower_cols[key]] = std
                    break
        if not rename:
            raise DataError(f"akshare 返回数据无可用列: {list(df.columns)}")
        out = df.rename(columns=rename)
        if "time" in out.columns:
            out["time"] = self._to_unix_seconds(out["time"])
        return out

    # ── 拉取：期货 ─────────────────────────────────────────────────────────

    def _fetch_futures(self, ak: object, code: str, tf: str) -> pd.DataFrame:
        if tf in _AK_MINUTE:
            return ak.futures_zh_minute_sina(symbol=code, period=_AK_MINUTE[tf])  # type: ignore[attr-defined]
        if tf in _AK_DAILY:
            return ak.futures_zh_daily_sina(symbol=code)  # type: ignore[attr-defined]
        raise DataError(f"akshare 期货不支持周期: {tf}")

    # ── 拉取：A 股 ─────────────────────────────────────────────────────────

    @staticmethod
    def _sina_code(code: str) -> str:
        """把 6 位 A 股代码补上新浪交易所前缀（``sh`` / ``sz`` / ``bj``）。"""
        c = (code or "").strip().lower()
        if c.startswith(("sh", "sz", "bj")):
            return c
        if c.startswith("6"):
            return "sh" + c
        if c.startswith(("0", "3")):
            return "sz" + c
        if c.startswith(("4", "8")):
            return "bj" + c
        return c

    def _fetch_equity(self, ak: object, code: str, tf: str) -> pd.DataFrame:
        """A 股拉取：首选东财 ``stock_zh_a_hist``；不可达时回退新浪日线。

        东财接口（``push2his.eastmoney.com``）在部分网络下不通，而新浪源与期货
        同源、可达性更好，故日线口径加一条新浪 ``stock_zh_a_daily`` 兜底。
        """
        try:
            if tf in _AK_EQUITY_PERIOD:
                end = _dt.date.today().strftime("%Y%m%d")
                return ak.stock_zh_a_hist(  # type: ignore[attr-defined]
                    symbol=code,
                    period=_AK_EQUITY_PERIOD[tf],
                    start_date=_AK_EQUITY_START,
                    end_date=end,
                    adjust="qfq",
                )
            if tf in _AK_EQUITY_MINUTE:
                return ak.stock_zh_a_hist_min_em(  # type: ignore[attr-defined]
                    symbol=code,
                    period=_AK_EQUITY_MINUTE[tf],
                    adjust="qfq",
                )
            raise DataError(
                f"akshare A 股不支持周期: {tf}（支持 "
                f"{sorted(_AK_EQUITY_PERIOD)} / {sorted(_AK_EQUITY_MINUTE)}）"
            )
        except DataError:
            raise
        except Exception:
            # 东财不可达 → 回退新浪日线（仅日线口径可兜底）
            if tf != "D1":
                raise
            return ak.stock_zh_a_daily(  # type: ignore[attr-defined]
                symbol=self._sina_code(code), adjust="qfq"
            )

    # ── 接口 ───────────────────────────────────────────────────────────────

    def fetch_full(self, symbol: str, timeframe: str) -> pd.DataFrame:
        if not self.is_available():
            raise DataError("akshare 不可用：未安装 akshare")
        try:
            import akshare as ak
        except ImportError as exc:
            raise DataError("未安装 akshare") from exc

        tf = timeframe.upper()
        # 调用方传入的 symbol 优先，否则回退构造期默认品种
        code = (symbol or "").strip() or self._symbol

        try:
            if self._market == "futures":
                raw = self._fetch_futures(ak, code, tf)
            else:
                raw = self._fetch_equity(ak, code, tf)
        except DataError:
            raise
        except Exception as exc:
            raise DataError(
                f"akshare 拉取失败: {self._market} {code} {timeframe}（{exc}）"
            ) from exc

        if raw is None or len(raw) == 0:
            raise DataError(f"akshare 无数据：{self._market} {code} {timeframe}")

        normalized = self._normalize_columns(raw)
        return self._finalize(normalized)
