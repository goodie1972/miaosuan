# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for full license text.

"""MetaTrader5 直连数据获取器（``[live]`` extra，仅 Windows）。

与 :mod:`~miaosuan.data.fetchers.mt4_bridge` 的 EA 桥不同，本 fetcher 通过官方
``MetaTrader5`` Python 包直连**本机 MT5 终端**（路线图 R1：神机本地库仍是生产
首选，网络下载为灾备，MT5 为第三通道）。

时间口径（R1.2，沿用 ``mt4_bridge`` 的 UTC 偏移探测方案）：MT5 返回的 bar
时间是**经纪商服务器时钟**的 Unix 秒标签，随经纪商时区（EET 类夏季 +3 /
冬季 +2）浮动，必须实测而非硬编码——``服务器当前时刻 - 本机 UTC 时刻``，
四舍五入到 0.5h 并缓存。输出口径由 ``time_base`` 控制（``utc`` 默认 /
``broker`` / ``shanghai``），与 :mod:`~miaosuan.data.panel` 的 UTC epoch 秒
契约对齐；缺省 ``utc`` 时即使经纪商终端直接返回 UTC（实测偏移 ≈ 0），换算
自动退化为恒等映射，无需按经纪商区分配置。

设计约束（同 :mod:`~miaosuan.data.fetchers.base`）：

* **禁止**读取环境变量——终端路径 / 凭据一律经构造参数注入；全部缺省
  ``None`` 时 ``initialize()`` 附着到**已登录的运行中终端**（最常见的本机用法）；
* ``MetaTrader5`` 包**方法内惰性导入**并包裹 ``try/except ImportError``，
  缺依赖（非 Windows / 未装 ``miaosuan[live]``）时 ``is_available()`` 优雅
  返回 ``False``，保证 ``acquisition.py`` 在缺依赖环境仍可正常 import；
* 每次取数 ``initialize()`` → 取数 → ``shutdown()``，不跨调用持有终端状态。
"""

from __future__ import annotations

import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import pandas as pd

from ...errors import DataError
from .base import BaseFetcher

__all__ = ["MT5Fetcher"]

#: 妙算周期标识 → 每周期秒数（增量拉取根数估算用；MN1 按 30 天近似）。
_TF_SECONDS: dict[str, int] = {
    "M1": 60,
    "M5": 300,
    "M15": 900,
    "M30": 1800,
    "H1": 3600,
    "H4": 14400,
    "D1": 86400,
    "W1": 604800,
    "MN1": 2_592_000,
}

#: ``time_base`` 可选值 -> 输出时区相对 UTC 的固定偏移（小时）。
_TZ_OFFSET_HOURS: dict[str, float] = {"utc": 0.0, "shanghai": 8.0}
_TIME_BASES = ("broker", "utc", "shanghai")

#: 全量拉取的请求上限。终端对超过本地可用深度的请求返回**空数组**（实测：
#: 5 / 100 / 1000 / 65536 正常返回，100000+ 返回 0），故全量拉取不能一把梭，
#: 需要「倍增上探 + 二分回退」找到实际可得的最大深度（见 :meth:`_copy_full`）。
_FULL_MAX_BARS = 200_000
#: 深度探测起点（终端再浅也不至于连 512 根都没有；更浅走逐级回退）。
_PROBE_START = 512


class MT5Fetcher(BaseFetcher):
    """MetaTrader5 直连 fetcher（委托官方 ``MetaTrader5`` 包）。

    Args:
        terminal_path: MT5 终端 terminal64.exe 路径；``None`` = 附着运行中的默认终端。
        login: 交易账号；仅自动登录时需要（附着已登录终端则全部留空）。
        password: 交易密码（与 ``login`` 配套）。
        server: 交易服务器名（与 ``login`` 配套）。
        time_base: 输出时间基准，``"utc"``（默认，符合 panel 契约）/
            ``"broker"``（保留经纪商时钟口径）/ ``"shanghai"``。
        utc_offset_hours: 经纪商时钟相对 UTC 的偏移（小时）。``None`` 时首次
            取数用 ``symbol_info_tick().time`` 实测并缓存；显式传入可跳过实测
            （测试 / 已知经纪商固定时区的场景）。
    """

    source_name = "mt5"

    def __init__(
        self,
        *,
        terminal_path: str | None = None,
        login: int | None = None,
        password: str | None = None,
        server: str | None = None,
        time_base: str = "utc",
        utc_offset_hours: float | None = None,
    ) -> None:
        if time_base not in _TIME_BASES:
            raise DataError(
                f"非法 time_base: {time_base!r}",
                context={"allowed": list(_TIME_BASES)},
            )
        self._terminal_path = terminal_path
        self._login = login
        self._password = password
        self._server = server
        self._time_base = time_base
        self._offset_cfg = utc_offset_hours
        self._offset_cache: float | None = None

    # ── 契约方法 ──────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """依赖就绪即视为可用（终端是否在运行推迟到取数时判定并如实报错）。"""
        if sys.platform != "win32":
            return False
        try:
            import MetaTrader5  # noqa: F401
        except ImportError:
            return False
        return True

    def describe(self) -> str:
        if not self.is_available():
            return "MT5: (缺 MetaTrader5，pip install 'miaosuan[live]'，仅 Windows)"
        cred = f", login={self._login}" if self._login is not None else ""
        return f"MT5 直连 (time_base={self._time_base}{cred})"

    def supported_timeframes(self) -> list[str]:
        return sorted(_TF_SECONDS)

    def fetch_full(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """全量拉取（深度自适应：倍增上探 + 二分回退，见 :meth:`_copy_full`）。"""
        self._check_symbol(symbol)
        with self._session() as mt5:
            self._ensure_symbol(mt5, symbol)
            rates = self._copy_full(mt5, symbol, self._tf_const(mt5, timeframe))
            offset = self._resolve_offset_hours(mt5, symbol)
            return self._to_frame(rates, symbol, timeframe, since_ts=None, offset_hours=offset)

    def fetch_incremental(self, symbol: str, timeframe: str, since_ts: int) -> pd.DataFrame:
        """增量拉取：按 ``(现在 - since_ts) / 周期秒`` 估算根数，换算后过滤。

        需要拉取的根数只取决于 ``now - since_ts``（两端同乘同一经纪商偏移，
        差值与偏移无关）；``since_ts`` 过滤必须在**时间基准换算之后**——
        原始经纪商时间无法直接与 UTC 口径的 ``since_ts`` 比较（同 mt4_bridge）。
        """
        self._check_symbol(symbol)
        tf = (timeframe or "").upper()
        if tf not in _TF_SECONDS:
            raise DataError(
                f"MT5 不支持的周期: {timeframe}",
                context={"supported": sorted(_TF_SECONDS)},
            )
        need = int((time.time() - int(since_ts)) // _TF_SECONDS[tf]) + 4
        count = max(2, min(need, _FULL_MAX_BARS))
        with self._session() as mt5:
            self._ensure_symbol(mt5, symbol)
            rates = self._copy(mt5, symbol, self._tf_const(mt5, tf), count)
            if len(rates) == 0 and count > _PROBE_START:
                # 请求根数超过终端本地深度 → 同样返回空，退回全量探测后按 since 过滤
                rates = self._copy_full(mt5, symbol, self._tf_const(mt5, tf))
            offset = self._resolve_offset_hours(mt5, symbol)
            return self._to_frame(rates, symbol, tf, since_ts=since_ts, offset_hours=offset)

    def detect_utc_offset_hours(self, symbol: str) -> float:
        """实测经纪商时钟相对 UTC 的偏移（小时，0.5h 精度）。

        独立公开供诊断 / 画像配置参考；取数路径内部走
        :meth:`_resolve_offset_hours`（带配置与缓存）。
        """
        self._check_symbol(symbol)
        with self._session() as mt5:
            self._ensure_symbol(mt5, symbol)
            return self._resolve_offset_hours(mt5, symbol, force_measure=True)

    # ── 内部 ─────────────────────────────────────────────────────────────

    @staticmethod
    def _copy(mt5: Any, symbol: str, tf_const: int, count: int) -> np.ndarray:
        """``copy_rates_from_pos`` 的空安全包装（终端异常时返回 ``None``）。"""
        arr = mt5.copy_rates_from_pos(symbol, tf_const, 0, count)
        return arr if arr is not None else np.zeros(0)

    @staticmethod
    def _copy_full(mt5: Any, symbol: str, tf_const: int) -> np.ndarray:
        """全量拉取：倍增上探 + 二分回退，找到终端实际可得的最大深度。

        终端对超过本地深度的请求返回**空数组**而非部分数据（实测 65536 可得、
        100000+ 返回 0），一把梭大请求会拿不到任何数据。策略：

        1. 从 :data:`_PROBE_START` 起 ×4 倍增上探；返回数**少于**请求数 = 终端
           已给尽，直接收工；返回空 = 请求过深，转二分；
        2. 在「最后一个可得点」与「首个空点」之间二分，逼近最大可得深度。

        每次请求都是终端内存操作，开销可忽略（十几次调用换全量历史）。
        """
        best = np.zeros(0)
        lo = 0
        hi = _PROBE_START
        while hi <= _FULL_MAX_BARS:
            arr = MT5Fetcher._copy(mt5, symbol, tf_const, hi)
            if len(arr) == 0:
                break  # hi 过深 → 二分 (lo, hi)
            best = arr
            lo = hi
            if len(arr) < hi:
                return arr  # 终端已给尽（少于请求）→ 这就是全量
            hi *= 4
        if len(best) == 0:
            # 连探测起点都空 → 极浅缓存，逐级回退到底
            for small in (128, 32, 8):
                arr = MT5Fetcher._copy(mt5, symbol, tf_const, small)
                if len(arr) > 0:
                    return arr
            return best
        while hi - lo > 1:
            mid = (lo + hi) // 2
            arr = MT5Fetcher._copy(mt5, symbol, tf_const, mid)
            if len(arr) > 0:
                best = arr
                lo = mid
            else:
                hi = mid
        return best

    @staticmethod
    def _ensure_symbol(mt5: Any, symbol: str) -> None:
        """确保品种在 Market Watch 中（``copy_rates`` 只对已选品种返回数据）。

        首次连接时终端 Market Watch 往往只有默认品种集，直接 copy 会得到
        空数据且无从排查——实测踩过：``XAUUSD`` 存在但未选中，``copy_rates``
        返回空数组。这里显式 ``symbol_select``，失败时给出近似品种候选。
        """
        if mt5.symbol_select(symbol, True):
            return
        candidates: list[str] = []
        try:
            upper = symbol.upper()
            for info in mt5.symbols_get() or []:
                name = getattr(info, "name", "")
                if upper in name.upper() and name not in candidates:
                    candidates.append(name)
                    if len(candidates) >= 10:
                        break
        except Exception:  # noqa: BLE001 - 候选仅用于报错提示，失败不影响主错误
            pass
        hint = f"（近似品种: {candidates}）" if candidates else ""
        raise DataError(
            f"MT5 品种不可用: {symbol}{hint}",
            context={"symbol": symbol, "candidates": candidates},
        )

    @staticmethod
    def _check_symbol(symbol: str) -> None:
        if not (symbol or "").strip():
            raise DataError("MT5 取数需要非空品种名", context={"symbol": symbol})

    @staticmethod
    def _load_mt5() -> Any:
        try:
            import MetaTrader5 as mt5  # 仅 [live] extra 提供
        except ImportError as exc:
            raise DataError(
                "缺少 MetaTrader5 包: pip install 'miaosuan[live]'（仅 Windows）"
            ) from exc
        return mt5

    @staticmethod
    def _tf_const(mt5: Any, timeframe: str) -> int:
        tf = (timeframe or "").upper()
        if tf not in _TF_SECONDS:
            raise DataError(
                f"MT5 不支持的周期: {timeframe}",
                context={"supported": sorted(_TF_SECONDS)},
            )
        const = getattr(mt5, f"TIMEFRAME_{tf}", None)
        if const is None:
            raise DataError(
                f"MetaTrader5 包缺少周期常量 TIMEFRAME_{tf}",
                context={"timeframe": tf},
            )
        return int(const)

    @contextmanager
    def _session(self) -> Iterator[Any]:
        """初始化终端 → 交付 ``mt5`` 模块 → 保证 ``shutdown()``。

        凭据全部缺省时不传任何参数：``initialize()`` 附着到**已登录的运行中
        终端**（本机最常见用法）；传了 ``login`` 才附带 ``password`` / ``server``。
        """
        mt5 = self._load_mt5()
        kwargs: dict[str, Any] = {}
        if self._terminal_path:
            kwargs["path"] = self._terminal_path
        if self._login is not None:
            kwargs["login"] = int(self._login)
            if self._password:
                kwargs["password"] = self._password
            if self._server:
                kwargs["server"] = self._server
        if not mt5.initialize(**kwargs):
            raise DataError(
                f"MT5 终端初始化失败: {mt5.last_error()}（终端未运行或未登录?）",
                context={"terminal_path": self._terminal_path or "(默认)"},
            )
        try:
            yield mt5
        finally:
            mt5.shutdown()

    def _resolve_offset_hours(self, mt5: Any, symbol: str, *, force_measure: bool = False) -> float:
        """经纪商时钟相对 UTC 的偏移（小时）；配置 > 缓存 > 实测，测不到按 0。

        实测：``symbol_info_tick().time``（服务器时钟的"现在"）减本机 UTC
        ``time.time()``，四舍五入到 0.5h（EET 类时区夏 +3 / 冬 +2）。
        测不到（休市 tick 为空 / 品种未知）按 UTC 处理（偏移 0）——与
        ``mt4_bridge`` 同口径，宁缺勿错；经纪商时区非 0 且休市时拉到的历史
        数据时间会整体偏移，可通过显式传 ``utc_offset_hours`` 兜底。
        """
        if self._offset_cfg is not None:
            return self._offset_cfg
        if self._offset_cache is not None and not force_measure:
            return self._offset_cache
        offset = 0.0
        try:
            tick = mt5.symbol_info_tick(symbol)
            tick_time = getattr(tick, "time", None) if tick is not None else None
            if tick_time:
                offset = round((int(tick_time) - int(time.time())) / 3600.0 * 2) / 2
        except Exception:  # noqa: BLE001 - 探测失败按 0 处理（同 mt4_bridge）
            offset = 0.0
        self._offset_cache = offset
        return offset

    def _to_broker_delta(self, offset_hours: float) -> int:
        """输出时间基准相对经纪商时钟的秒数偏移（``output = raw + delta``）。"""
        if self._time_base == "broker":
            return 0
        return int(round((_TZ_OFFSET_HOURS[self._time_base] - offset_hours) * 3600))

    def _to_frame(
        self,
        rates: Any,
        symbol: str,
        timeframe: str,
        *,
        since_ts: int | None,
        offset_hours: float,
    ) -> pd.DataFrame:
        """把 ``copy_rates_*`` 的结构化数组转成统一契约 DataFrame（含时区换算）。"""
        df = pd.DataFrame(rates)
        if df is None or len(df) == 0:
            return self._finalize(pd.DataFrame())
        missing = [
            c
            for c in ("time", "open", "high", "low", "close", "tick_volume")
            if c not in df.columns
        ]
        if missing:
            raise DataError(
                f"MT5 返回数据缺少列: {missing}",
                context={"symbol": symbol, "timeframe": timeframe, "columns": list(df.columns)},
            )
        # volume 用 tick_volume（外汇 / 贵金属 CFD 无真实成交量，与 loader 契约一致）
        df["volume"] = df["tick_volume"].astype("float64")
        delta = self._to_broker_delta(offset_hours)
        df["time"] = (df["time"].astype("int64") + int(delta)).astype("int64")
        if since_ts is not None:
            # 必须在时间基准换算之后过滤：原始经纪商时间无法直接与 UTC 口径比较
            df = df[df["time"] > int(since_ts)]
        return self._finalize(df)
