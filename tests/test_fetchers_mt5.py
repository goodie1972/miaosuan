# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""MT5 直连 fetcher 测试（路线图 R1.5，**不依赖真实 MT5 终端**）。

``MetaTrader5`` 包以 ``sys.modules`` 假模块注入：``copy_rates_from_pos`` 返回
**结构化 numpy 数组**（与真实包同形态），时间偏移 / 增量根数 / 命名契约 /
缓存兜底全部离线断言，与机器是否装有 MT5 终端解耦（范式同
``test_fetchers.py`` 的 socket 层打桩）。

三条路线图验收断言（G1）：

* 联网（假终端在线）：首跑全量 → 追加新 bar → 二次拉取只取增量并合并去重；
* 断网（假终端掉线）：已有缓存时回退旧缓存，``STALE_FALLBACK`` 且如实带错误；
* 命名契约：``source="MT5"`` 落盘 ``{symbol}_{timeframe}_{source_type}.parquet``。
"""

from __future__ import annotations

import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from miaosuan.data.acquisition import (
    _FETCHER_TYPE_MAP,
    DataAcquisition,
    FetchStatus,
    TypedNetworkSource,
)
from miaosuan.data.fetchers import MT5Fetcher, all_network_sources
from miaosuan.errors import DataError

#: 与真实 ``copy_rates_from_pos`` 相同的字段形态（结构化数组）。
_RATES_DTYPE = np.dtype(
    [
        ("time", "<i8"),
        ("open", "<f8"),
        ("high", "<f8"),
        ("low", "<f8"),
        ("close", "<f8"),
        ("tick_volume", "<i8"),
        ("spread", "<i4"),
        ("real_volume", "<i8"),
    ]
)

_SERVER_OFFSET_HOURS = 2.0  # 假经纪商：服务器时钟 = UTC + 2h（EET 冬季口径）


class _FakeMT5Module(types.ModuleType):
    """假 MetaTrader5 模块：可注入 bar 存储、可切换"终端掉线"。

    ``copy_rates_from_pos(symbol, tf, start_pos, count)`` 只支持 ``start_pos=0``
    （本 fetcher 的唯一用法），返回存储 bar 的**末 count 根**。
    """

    TIMEFRAME_M1 = 1
    TIMEFRAME_M5 = 5
    TIMEFRAME_M15 = 15
    TIMEFRAME_M30 = 30
    TIMEFRAME_H1 = 16385
    TIMEFRAME_H4 = 16388
    TIMEFRAME_D1 = 16408
    TIMEFRAME_W1 = 32769
    TIMEFRAME_MN1 = 49153

    def __init__(
        self,
        store: dict[tuple[str, str], list[tuple]],
        *,
        server_offset_hours: float = _SERVER_OFFSET_HOURS,
        down: bool = False,
        known_symbols: tuple[str, ...] = ("XAUUSD",),
        max_depth: int | None = None,
    ) -> None:
        super().__init__("MetaTrader5")
        self._store = store
        self._offset = server_offset_hours
        self.down = down
        self.known_symbols = known_symbols
        # 复刻真实终端行为：请求超过本地深度的历史时返回**空数组**（实测
        # 65536 可得、100000+ 返回 0）；None = 不限制（有多少给多少）
        self.max_depth = max_depth
        self.init_kwargs: dict[str, Any] | None = None
        self.copy_calls: list[tuple[str, int, int, int]] = []
        self.select_calls: list[str] = []

    def initialize(self, **kwargs: Any) -> bool:
        self.init_kwargs = dict(kwargs)
        return not self.down

    def shutdown(self) -> None:
        return None

    def last_error(self) -> tuple[int, str]:
        return (10004, "fake terminal offline") if self.down else (0, "ok")

    def symbol_select(self, symbol: str, enable: bool) -> bool:
        self.select_calls.append(symbol)
        return symbol in self.known_symbols

    def symbols_get(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name=n) for n in self.known_symbols]

    def symbol_info_tick(self, symbol: str) -> SimpleNamespace | None:
        if self.down:
            return None
        # 服务器时钟的"现在"（真实包语义：tick.time 为服务器时钟 Unix 秒标签）
        return SimpleNamespace(time=int(time.time()) + int(self._offset * 3600))

    def copy_rates_from_pos(
        self, symbol: str, timeframe: int, start_pos: int, count: int
    ) -> np.ndarray:
        self.copy_calls.append((symbol, int(timeframe), int(start_pos), int(count)))
        if self.down:
            return np.zeros(0, dtype=_RATES_DTYPE)
        bars = self._store.get((symbol, "M30"), [])
        if start_pos != 0 or not bars:
            return np.zeros(0, dtype=_RATES_DTYPE)
        if self.max_depth is not None and count > self.max_depth:
            return np.zeros(0, dtype=_RATES_DTYPE)
        return np.array(bars[-count:], dtype=_RATES_DTYPE)


def make_m30_bars(n: int, *, end_utc: int) -> list[tuple]:
    """生成 n 根连续 M30 bar；``raw = utc + 服务器偏移``（经纪商时钟标签）。"""
    tf = 1800
    end_raw = end_utc + int(_SERVER_OFFSET_HOURS * 3600)
    rows: list[tuple] = []
    price = 2000.0
    for i in range(n):
        t = end_raw - (n - 1 - i) * tf
        price += 0.5
        rows.append((t, price, price + 1.0, price - 1.0, price + 0.2, 100 + i, 0, 0))
    return rows


@pytest.fixture
def cache_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """缓存目录隔离：kline 目录与缓存目录都指向 tmp（前者在目录列表里**排第一**，
    不隔离会把测试数据写进真实 K 线目录），不碰任何真实缓存。"""
    from miaosuan.data.acquisition import _cache_dirs_from_config
    from miaosuan.config import kline_data_dir
    monkeypatch.setattr("miaosuan.data.acquisition._cache_dirs_from_config", lambda: (tmp_path,))
    monkeypatch.setattr("miaosuan.config.kline_data_dir", lambda: str(tmp_path))
    return tmp_path


@pytest.fixture
def fake_mt5(monkeypatch: pytest.MonkeyPatch):
    """注入假 MetaTrader5 模块的工厂；返回 (install, 取最近一次安装的模块)。"""
    holder: dict[str, _FakeMT5Module] = {}

    def _install(store: dict[tuple[str, str], list[tuple]], **kw: Any) -> _FakeMT5Module:
        mod = _FakeMT5Module(store, **kw)
        monkeypatch.setitem(sys.modules, "MetaTrader5", mod)
        holder["mod"] = mod
        return mod

    holder["install"] = _install  # type: ignore[assignment]
    return holder


def _make_acquisition(fetcher: MT5Fetcher) -> DataAcquisition:
    return DataAcquisition(sources=[TypedNetworkSource(fetcher)])


# ── 验收断言一：命名契约 + 全量契约 ─────────────────────────────────────────


def test_fetch_full_naming_contract(fake_mt5: Any, cache_env: Path) -> None:
    """source="MT5" 落盘 {symbol}_{timeframe}_{source_type}.parquet（G1 命名契约）。"""
    end_utc = int(time.time())
    fake_mt5["install"]({("XAUUSD", "M30"): make_m30_bars(50, end_utc=end_utc)})
    fetcher = MT5Fetcher(utc_offset_hours=_SERVER_OFFSET_HOURS)
    acq = _make_acquisition(fetcher)

    res = acq.fetch("XAUUSD", "M30", source="MT5")

    assert res.status is FetchStatus.UPDATED
    assert res.source_used == "MT5"
    assert res.path.name == "XAUUSD_M30_MT5.parquet"
    assert res.path.parent == cache_env
    df = pd.read_parquet(res.path)
    assert len(df) == 50
    # time 已从经纪商时钟换算回 UTC 口径（panel 契约）
    assert int(df["time"].max()) == end_utc
    assert list(df.columns) == ["time", "open", "high", "low", "close", "volume", "tick_volume"]


# ── 验收断言二：联网增量（只拉新 bar、合并去重）────────────────────────────


def test_incremental_pulls_only_new_bars(fake_mt5: Any, cache_env: Path) -> None:
    end_utc = int(time.time())
    bars = make_m30_bars(50, end_utc=end_utc)
    mod = fake_mt5["install"]({("XAUUSD", "M30"): bars})
    acq = _make_acquisition(MT5Fetcher(utc_offset_hours=_SERVER_OFFSET_HOURS))

    first = acq.fetch("XAUUSD", "M30", source="MT5")
    assert first.status is FetchStatus.UPDATED
    full_count = mod.copy_calls[0][3]

    # 追加 3 根新 bar（时间延续），二次拉取应只取尾部少量根数
    price = float(bars[-1][4])
    for i in range(1, 4):
        t = int(bars[-1][0]) + 1800 * i
        price += 0.5
        bars.append((t, price, price + 1.0, price - 1.0, price + 0.2, 100 + i, 0, 0))

    second = acq.fetch("XAUUSD", "M30", source="MT5")

    assert second.status is FetchStatus.UPDATED
    assert second.path == first.path
    inc_count = mod.copy_calls[-1][3]
    assert inc_count < full_count, "增量拉取根数必须远小于全量"
    df = pd.read_parquet(second.path)
    assert len(df) == 53, "增量 3 根必须并入缓存"
    assert df["time"].is_monotonic_increasing and not df["time"].duplicated().any()
    # 增量只含 time > 旧 max(time) 的行（换算后过滤，绝无重复 bar）
    assert int(df["time"].max()) == int(bars[-1][0]) - int(_SERVER_OFFSET_HOURS * 3600)


def test_up_to_date_when_no_new_bars(fake_mt5: Any, cache_env: Path) -> None:
    fake_mt5["install"]({("XAUUSD", "M30"): make_m30_bars(50, end_utc=int(time.time()))})
    acq = _make_acquisition(MT5Fetcher(utc_offset_hours=_SERVER_OFFSET_HOURS))

    assert acq.fetch("XAUUSD", "M30", source="MT5").status is FetchStatus.UPDATED
    assert acq.fetch("XAUUSD", "M30", source="MT5").status is FetchStatus.UP_TO_DATE


# ── 验收断言三：断网走本地缓存 ─────────────────────────────────────────────


def test_offline_falls_back_to_cache(fake_mt5: Any, cache_env: Path) -> None:
    mod = fake_mt5["install"]({("XAUUSD", "M30"): make_m30_bars(50, end_utc=int(time.time()))})
    acq = _make_acquisition(MT5Fetcher(utc_offset_hours=_SERVER_OFFSET_HOURS))

    first = acq.fetch("XAUUSD", "M30", source="MT5")
    assert first.status is FetchStatus.UPDATED
    before = pd.read_parquet(first.path)

    mod.down = True  # 终端掉线：initialize() 返回 False
    stale = acq.fetch("XAUUSD", "M30", source="MT5")

    assert stale.status is FetchStatus.STALE_FALLBACK
    assert stale.path == first.path
    assert stale.errors, "STALE_FALLBACK 必须如实带上失败原因"
    pd.testing.assert_frame_equal(pd.read_parquet(stale.path), before)


def test_terminal_offline_without_cache_raises(fake_mt5: Any, cache_env: Path) -> None:
    fake_mt5["install"]({("XAUUSD", "M30"): make_m30_bars(50, end_utc=int(time.time()))}, down=True)
    acq = _make_acquisition(MT5Fetcher(utc_offset_hours=_SERVER_OFFSET_HOURS))

    with pytest.raises(DataError, match="无法获取"):
        acq.fetch("XAUUSD", "M30", source="MT5")


# ── 时间口径（R1.2）：实测偏移换算 / broker 直通 ───────────────────────────


def test_measured_utc_offset_conversion(fake_mt5: Any, cache_env: Path) -> None:
    """utc_offset_hours=None：用 tick 实测 +2h 偏移，输出换算回 UTC。"""
    end_utc = int(time.time())
    fake_mt5["install"]({("XAUUSD", "M30"): make_m30_bars(10, end_utc=end_utc)})
    acq = _make_acquisition(MT5Fetcher())  # 不显式给偏移，走实测路径

    res = acq.fetch("XAUUSD", "M30", source="MT5")
    df = pd.read_parquet(res.path)
    assert int(df["time"].max()) == end_utc, "实测偏移 +2h 必须被换算掉"


def test_broker_time_base_no_conversion(fake_mt5: Any, cache_env: Path) -> None:
    end_utc = int(time.time())
    fake_mt5["install"]({("XAUUSD", "M30"): make_m30_bars(10, end_utc=end_utc)})
    fetcher = MT5Fetcher(time_base="broker", utc_offset_hours=_SERVER_OFFSET_HOURS)
    res = _make_acquisition(fetcher).fetch("XAUUSD", "M30", source="MT5")

    df = pd.read_parquet(res.path)
    assert int(df["time"].max()) == end_utc + int(_SERVER_OFFSET_HOURS * 3600)


def test_detect_utc_offset_hours(fake_mt5: Any, cache_env: Path) -> None:
    fake_mt5["install"]({("XAUUSD", "M30"): make_m30_bars(5, end_utc=int(time.time()))})
    assert MT5Fetcher().detect_utc_offset_hours("XAUUSD") == pytest.approx(2.0)


# ── 可用性与注册 ───────────────────────────────────────────────────────────


def test_is_available_platform_and_dependency(
    fake_mt5: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_mt5["install"]({})  # 依赖就绪（假包已注入）
    assert MT5Fetcher().is_available() is True

    monkeypatch.setattr(sys, "platform", "linux")
    assert MT5Fetcher().is_available() is False, "非 Windows 平台必须优雅判不可用"

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "MetaTrader5", None)  # None → import 抛 ImportError
    assert MT5Fetcher().is_available() is False


def test_unsupported_timeframe_raises(fake_mt5: Any) -> None:
    fake_mt5["install"]({})
    fetcher = MT5Fetcher()
    with pytest.raises(DataError, match="不支持的周期"):
        fetcher.fetch_full("XAUUSD", "H2")
    with pytest.raises(DataError, match="不支持的周期"):
        fetcher.fetch_incremental("XAUUSD", "H2", since_ts=0)


def test_symbol_not_in_market_watch_selects_first(fake_mt5: Any, cache_env: Path) -> None:
    """真实终端踩过的坑：品种存在但不在 Market Watch，copy 返回空——须先 select。"""
    end_utc = int(time.time())
    mod = fake_mt5["install"](
        {("XAUUSD", "M30"): make_m30_bars(10, end_utc=end_utc)},
        known_symbols=("XAUUSD", "XAUEUR", "GOLD"),
    )
    res = _make_acquisition(MT5Fetcher(utc_offset_hours=_SERVER_OFFSET_HOURS)).fetch(
        "XAUUSD", "M30", source="MT5"
    )
    assert res.status is FetchStatus.UPDATED
    assert mod.select_calls == ["XAUUSD"]


def test_unknown_symbol_raises_with_candidates(fake_mt5: Any, cache_env: Path) -> None:
    fake_mt5["install"]({}, known_symbols=("XAUUSDm", "GOLD", "XAUEUR"))
    fetcher = MT5Fetcher()
    with pytest.raises(DataError, match="XAUUSDm") as excinfo:
        fetcher.fetch_full("XAUUSD", "M30")
    assert "近似品种" in str(excinfo.value)


def test_full_fetch_probes_depth_when_large_request_empty(fake_mt5: Any, cache_env: Path) -> None:
    """真实终端行为：请求超过本地深度的历史返回**空数组**——须倍增上探+二分回退。"""
    end_utc = int(time.time())
    fake_mt5["install"](
        {("XAUUSD", "M30"): make_m30_bars(3000, end_utc=end_utc)},
        max_depth=1000,  # 请求 > 1000 一律返回空
    )
    res = _make_acquisition(MT5Fetcher(utc_offset_hours=_SERVER_OFFSET_HOURS)).fetch(
        "XAUUSD", "M30", source="MT5"
    )

    assert res.status is FetchStatus.UPDATED, "深度受限时必须探测到最大可得深度，而非报空数据"
    df = pd.read_parquet(res.path)
    assert len(df) == 1000, "应拿到终端最大可得深度（1000 根），而非 0"


@pytest.mark.skipif(sys.platform != "win32", reason="MetaTrader5 仅 Windows")
def test_registered_as_default_source() -> None:
    assert _FETCHER_TYPE_MAP["MT5Fetcher"] == "MT5"
    assert any(isinstance(s, MT5Fetcher) for s in all_network_sources())
