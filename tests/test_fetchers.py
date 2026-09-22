# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""数据获取器测试（``data/fetchers`` + acquisition 接入）。

设计目标：
* 神机库读取用「**真实库优先、合成库兜底**」的 fixture 验证（列契约 / 升序 /
  无重复 / 只读）——无论如何都会真跑一遍断言，不再因本机无 DB 而**静默 0 覆盖**；
* 网络 fetcher 的 TCP 可达性探测一律用 monkeypatch 在 **socket 层**打桩，
  **不发任何真实网络请求**，与机器网络状态解耦（范式同 ``test_dukascopy_probe.py``）；
* OKX 恒可用；Dukascopy / Binance 的可用性由桩决定，断言因此稳定可复现。
"""

from __future__ import annotations

import os
import socket
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from miaosuan.config import (
    DataAcquisitionConfig,
    _default_shenji_db_path,
    data_acquisition_config,
)
from miaosuan.data.acquisition import DataAcquisition, ShenjiLocalSource
from miaosuan.data.fetchers import (
    AkshareFetcher,
    BaseFetcher,
    DukascopyFetcher,
    OkxFetcher,
    ShenjiDBFetcher,
    TradingViewFetcher,
    TqsdkFetcher,
    list_network_sources,
)


# ── 装置一：把 TCP 探测固定为「可达」──────────────────────────────────────


class _FakeSock:
    """假 socket：探测桩只需能被 ``close()``。"""

    def close(self) -> None:  # pragma: no cover - 桩
        return None


@pytest.fixture
def tcp_reachable(monkeypatch: pytest.MonkeyPatch) -> None:
    """把 ``socket`` 探测固定为**可达**，不发起任何真实连接。

    Dukascopy / Binance 的 ``is_available()`` 会做 TCP 探测；放任它真连的话，
    离线机器要空等 ``probe_timeout`` 秒，网络抖动时结果还会变——两条都会让
    本文件的用例偶发失败。统一在 socket 层打桩后，断言只取决于桩的返回值。
    """
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda host, port, family=0, socktype=0, proto=0, flags=0: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("203.0.113.7", port))
        ],
    )
    monkeypatch.setattr(
        socket,
        "create_connection",
        lambda address, timeout=None, **kwargs: _FakeSock(),
    )


# ── 装置二：神机库（真实优先，合成兜底）──────────────────────────────────


def _real_shenji_db() -> str:
    """返回神机库真实路径（不存在则空串）。"""
    return _default_shenji_db_path()


def _make_synthetic_shenji_db(directory: Path) -> str:
    """在临时目录造一个**合成**神机库（**绝不触碰**真实 ``market_data.db``）。

    表结构与神机落库一致：
    ``ohlcv(timeframe, timestamp, open, high, low, close, volume)``。
    """
    db_path = str(directory / "synthetic_market_data.db")
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "CREATE TABLE ohlcv ("
            "timeframe TEXT, timestamp INTEGER, open REAL, high REAL, "
            "low REAL, close REAL, volume REAL)"
        )
        base = 1_700_000_000
        conn.executemany(
            "INSERT INTO ohlcv VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    "H1",
                    base + i * 3600,
                    100.0 + i,
                    101.0 + i,
                    99.0 + i,
                    100.5 + i,
                    10.0 + i,
                )
                for i in range(10)
            ],
        )
        conn.commit()
    finally:
        conn.close()
    return db_path


@pytest.fixture
def shenji_db(tmp_path: Path) -> str:
    """真实神机库优先；不存在时用合成库兜底。

    早期版本在无 DB 的机器上直接 ``pytest.skip``，这些用例于是**静默 0 覆盖**——
    看起来绿，其实一条断言都没跑。兜底后无论有没有真实库都会真验证一遍。
    """
    real = _real_shenji_db()
    if real:
        return real
    return _make_synthetic_shenji_db(tmp_path)


# ── 用例 ──────────────────────────────────────────────────────────────────


def test_shenji_db_reads(shenji_db: str) -> None:
    f = ShenjiDBFetcher(shenji_db)
    assert f.is_available() is True

    mtime_before = os.path.getmtime(shenji_db)
    df = f.fetch_full("XAUUSD", "H1")
    mtime_after = os.path.getmtime(shenji_db)
    # 只读：读取不应改变文件 mtime
    assert mtime_before == mtime_after

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    for col in ("time", "open", "high", "low", "close", "volume"):
        assert col in df.columns
    # 升序且无重复时间戳
    assert df["time"].is_monotonic_increasing
    assert df["time"].nunique() == len(df)
    # 增量：只取后半段
    last = int(df["time"].iloc[-1])
    mid = int(df["time"].iloc[len(df) // 2])
    inc = f.fetch_incremental("XAUUSD", "H1", mid)
    assert not inc.empty
    assert inc["time"].min() > mid
    assert inc["time"].max() == last


def test_network_sources_instantiate(tcp_reachable: None) -> None:
    classes = [
        TradingViewFetcher,
        AkshareFetcher,
        TqsdkFetcher,
        OkxFetcher,
        DukascopyFetcher,
    ]
    for cls in classes:
        inst = cls()
        avail = inst.is_available()
        assert isinstance(avail, bool)
        # 不应抛异常
        _ = inst.describe()


def test_list_network_sources_returns_list(tcp_reachable: None) -> None:
    sources = list_network_sources()
    assert isinstance(sources, list)
    for s in sources:
        assert isinstance(s, BaseFetcher)
        # 契约：list_network_sources() 只返回当前环境可用的来源
        assert s.is_available() is True
    kinds = {type(s).__name__ for s in sources}
    # 探测已被桩固定为可达，故这三个来源必定在列（与真实网络状态无关）。
    # 早期版本此处无条件断言 Dukascopy 存在，而它改做 TCP 探测后离线机器会
    # 把它过滤掉 → 用例随网络状态漂移；现在由桩锁定，断言稳定可复现。
    assert "OkxFetcher" in kinds
    assert "DukascopyFetcher" in kinds
    assert "BinanceFetcher" in kinds


def test_acquisition_registers_shenji(shenji_db: str) -> None:
    # 显式注入来源（而非依赖本机默认配置）后，无真实 DB 的机器上也能验证
    # 「注册 + 可用性 + 真能读出库」这一整条链路。
    acq = DataAcquisition(sources=[ShenjiLocalSource(shenji_db)])
    sources = acq.list_sources()
    names = [s["name"] for s in sources]
    assert "ShenjiLocalSource" in names
    shenji = next(s for s in sources if s["name"] == "ShenjiLocalSource")
    assert shenji["available"] is True
    # 确认 ShenjiLocalSource 真实可用并能读库
    src = acq._sources[names.index("ShenjiLocalSource")]
    assert src.is_available()
    df = src.fetch_full("XAUUSD", "H1")
    assert not df.empty


def test_data_acquisition_config_defaults() -> None:
    cfg = data_acquisition_config(env={})
    assert isinstance(cfg, DataAcquisitionConfig)
    assert cfg.data_source == ""
    assert cfg.dukascopy_user == ""
    assert cfg.dukascopy_password == ""
