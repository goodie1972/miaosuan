# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for full license text.

"""数据获取器测试（``data/fetchers`` + acquisition 接入）。

设计目标：
* 真实神机库读取在 DB 存在时验证（列契约 / 升序 / 无重复 / 只读）；
* 网络 fetcher 在重依赖缺失环境下优雅跳过（``is_available()`` 返回 bool 不抛异常）；
* OKX 走公开接口恒可用；Dukascopy / Binance 走公开接口但会做 TCP 可达性探测，
  主机不可达时 ``is_available()`` 返回 ``False``（测试须对网络状态保持中立）。
"""

from __future__ import annotations

import os

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


def _real_shenji_db() -> str:
    """返回神机库真实路径（不存在则空串）。"""
    return _default_shenji_db_path()


def test_shenji_db_reads_real() -> None:
    db = _real_shenji_db()
    if not db:
        pytest.skip("神机本地库不存在，跳过真实读取测试")
    f = ShenjiDBFetcher(db)
    assert f.is_available() is True

    mtime_before = os.path.getmtime(db)
    df = f.fetch_full("XAUUSD", "H1")
    mtime_after = os.path.getmtime(db)
    # 只读：读取不应改变文件 mtime
    assert mtime_before == mtime_after

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    for col in ("time", "open", "high", "low", "close", "volume"):
        assert col in df.columns
    # 升序且无重复时间戳
    assert df["time"].is_monotonic_increasing
    assert df["time"].nunique() == len(df)
    # H1 行存在
    assert len(df) > 0
    # 增量：只取后半段
    last = int(df["time"].iloc[-1])
    mid = int(df["time"].iloc[len(df) // 2])
    inc = f.fetch_incremental("XAUUSD", "H1", mid)
    assert not inc.empty
    assert inc["time"].min() > mid
    assert inc["time"].max() == last


def test_network_sources_instantiate() -> None:
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


def test_list_network_sources_returns_list() -> None:
    sources = list_network_sources()
    assert isinstance(sources, list)
    for s in sources:
        assert isinstance(s, BaseFetcher)
        # 契约：list_network_sources() 只返回当前环境可用的来源
        assert s.is_available() is True
    kinds = {type(s).__name__ for s in sources}
    # OKX 公开 REST 接口恒可用（is_available() 无条件返回 True）
    assert "OkxFetcher" in kinds
    # Dukascopy 虽走公开免登录接口，但自「TCP 可达性探测」改动后其
    # is_available() 会在主机不可达时返回 False，从而被 list_network_sources()
    # 正确过滤（Binance 自始即如此）。故按可达性一致性断言，使离线/沙箱与
    # 联网环境都能稳定通过。
    duk = DukascopyFetcher()
    assert ("DukascopyFetcher" in kinds) == duk.is_available()


def test_acquisition_registers_shenji() -> None:
    db = _real_shenji_db()
    if not db:
        pytest.skip("神机本地库不存在，跳过")
    acq = DataAcquisition()
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
