# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""数据加载器单测（M6）—— 复刻 AM ``data_pipeline`` 语义。

覆盖：成交量列名（``tick_volume`` 优先）、``time<1e7 → ×1000``、排序去重（``keep="last"``）、
``float32`` / ``int64``、缺失值保留为 ``np.nan``、文件名契约推断、错误分支。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from miaosuan.data import loader
from miaosuan.data.loader import (
    TIME_SCALE_THRESHOLD,
    infer_symbol_timeframe,
    load,
    load_csv,
    normalize_timeframe,
    panel_from_frame,
)
from miaosuan.errors import DataError


def _frame(rows: list[dict[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _basic_rows(n: int = 5) -> list[dict[str, float]]:
    return [
        {
            "time": 1_700_000_000 + i * 3600,
            "open": 100.0 + i,
            "high": 101.0 + i,
            "low": 99.0 + i,
            "close": 100.5 + i,
            "volume": 10.0 + i,
        }
        for i in range(n)
    ]


def test_basic_load_shape_and_dtype() -> None:
    p = panel_from_frame(_frame(_basic_rows(5)), symbols=("XAUUSD",), timeframe="H1")
    assert p.open.shape == (1, 5)
    assert p.open.dtype == np.float32
    assert p.time.dtype == np.int64
    assert p.n_bars == 5
    assert p.meta["adjustment_mode"] == "none"


def test_tick_volume_preferred() -> None:
    df = _frame(_basic_rows(4))
    df["tick_volume"] = [100, 200, 300, 400]
    p = panel_from_frame(df, symbols=("XAUUSD",))
    assert p.meta["volume_column"] == "tick_volume"
    assert p.volume[0, -1] == np.float32(400.0)


def test_explicit_volume_column() -> None:
    df = _frame(_basic_rows(3))
    df["tick_volume"] = [1, 2, 3]
    p = panel_from_frame(df, symbols=("X",), volume_column="volume")
    assert p.meta["volume_column"] == "volume"
    assert p.volume[0, 0] == np.float32(10.0)


def test_time_scale_guard_multiplies_by_1000() -> None:
    rows = _basic_rows(3)
    for i, r in enumerate(rows):
        r["time"] = 1_600_000 + i * 3600  # < 1e7 → 视为秒/1000
    p = panel_from_frame(_frame(rows), symbols=("X",))
    # max(time)=1_607_200 < 1e7 → auto 修复为 ×1000；最小值为 1_600_000*1000
    assert int(p.time.min()) == 1_600_000 * 1000
    assert int(p.time.min()) > TIME_SCALE_THRESHOLD


def test_time_unit_ms_explicit() -> None:
    rows = _basic_rows(3)
    for i, r in enumerate(rows):
        r["time"] = (1_700_000_000 + i * 3600) * 1000  # 毫秒
    p = panel_from_frame(_frame(rows), symbols=("X",), time_unit="ms")
    assert int(p.time[0]) == 1_700_000_000


def test_bad_time_unit_rejected() -> None:
    with pytest.raises(DataError):
        panel_from_frame(_frame(_basic_rows(3)), symbols=("X",), time_unit="nanoseconds")


def test_sort_and_dedup_keep_last() -> None:
    rows = [
        {"time": 1_700_000_7200, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0},
        {"time": 1_700_000_0000, "open": 2.0, "high": 2.0, "low": 2.0, "close": 2.0, "volume": 2.0},
        {"time": 1_700_000_7200, "open": 9.0, "high": 9.0, "low": 9.0, "close": 9.0, "volume": 9.0},
    ]
    p = panel_from_frame(_frame(rows), symbols=("X",))
    assert p.n_bars == 2, "重复时间戳应去重"
    assert np.all(np.diff(p.time) > 0), "应升序排列"
    # keep="last"：时间 7200 的组保留第二条（open=9）
    assert p.open[0, -1] == np.float32(9.0)


def test_nan_preserved_not_zero_filled() -> None:
    rows = _basic_rows(3)
    rows[1]["close"] = float("nan")
    p = panel_from_frame(_frame(rows), symbols=("X",))
    assert np.isnan(p.close[0, 1])
    assert not np.isnan(p.close[0, 0])


def test_infer_symbol_timeframe() -> None:
    assert infer_symbol_timeframe("XAUUSD_H1.parquet") == ("XAUUSD", "H1")
    assert infer_symbol_timeframe("BTCUSDT_60min.csv") == ("BTCUSDT", "H1")
    assert infer_symbol_timeframe("weird.parquet") == ("weird", "")


def test_normalize_timeframe_alias() -> None:
    assert normalize_timeframe("1h") == "H1"
    assert normalize_timeframe("5min") == "M5"
    assert normalize_timeframe("daily") == "D1"
    assert normalize_timeframe("zzz") == "ZZZ"


def test_load_csv_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "XAUUSD_H1.csv"
    _frame(_basic_rows(6)).to_csv(path, index=False)
    p = load(path)
    assert p.n_bars == 6
    assert p.symbols == ("XAUUSD",)
    assert p.timeframe == "H1"
    assert p.meta["format"] == "csv"


def test_load_missing_file() -> None:
    with pytest.raises(DataError):
        load_csv("does_not_exist_12345.csv")


def test_load_unknown_extension(tmp_path: Path) -> None:
    path = tmp_path / "XAUUSD_H1.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(DataError):
        load(path)


def test_missing_columns_rejected() -> None:
    df = pd.DataFrame({"time": [1_700_000_000], "open": [1.0]})
    with pytest.raises(DataError):
        panel_from_frame(df, symbols=("X",))


def test_no_volume_column_rejected() -> None:
    df = _frame(_basic_rows(2)).drop(columns=["volume"])
    with pytest.raises(DataError):
        panel_from_frame(df, symbols=("X",))


def test_empty_frame_rejected() -> None:
    with pytest.raises(DataError):
        panel_from_frame(pd.DataFrame(), symbols=("X",))


def test_infer_and_normalize_exposed() -> None:
    assert loader.TIME_SCALE_THRESHOLD == 10_000_000
