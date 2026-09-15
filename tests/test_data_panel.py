# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""``Panel`` 数据契约单测（M6）。

覆盖：形状/类型校验、``[N,T]`` 契约、``slice_view``、``to_raw_dict``、指纹确定性与敏感性。
"""

from __future__ import annotations

import numpy as np
import pytest

from miaosuan.data.panel import PANEL_FIELDS, Panel
from miaosuan.errors import DataError


def _panel(n: int = 2, t: int = 8) -> Panel:
    rng = np.random.default_rng(7)
    fields = {name: rng.normal(100, 1, (n, t)).astype(np.float32) for name in PANEL_FIELDS}
    time = (np.arange(t, dtype=np.int64) + 1) * 3600
    return Panel.from_arrays(
        fields,
        time,
        symbols=tuple(f"S{i}" for i in range(n)),
        timeframe="H1",
        market_profile_name="FOREX_XAUUSD",
    )


def test_panel_shapes_and_props() -> None:
    p = _panel(3, 10)
    assert p.n_symbols == 3
    assert p.n_bars == 10
    assert set(p.fields) == set(PANEL_FIELDS)
    assert all(p.fields[name].dtype == np.float32 for name in PANEL_FIELDS)
    assert p.time.dtype == np.int64
    assert p.market_profile_name == "FOREX_XAUUSD"


def test_panel_rejects_shape_mismatch() -> None:
    fields = {name: np.zeros((1, 8), dtype=np.float32) for name in PANEL_FIELDS}
    fields["close"] = np.zeros((1, 7), dtype=np.float32)
    with pytest.raises(DataError):
        Panel.from_arrays(fields, np.arange(8, dtype=np.int64) * 3600, symbols=("A",))


def test_panel_rejects_non_increasing_time() -> None:
    fields = {name: np.zeros((1, 4), dtype=np.float32) for name in PANEL_FIELDS}
    bad_time = np.array([0, 2, 1, 3], dtype=np.int64)
    with pytest.raises(DataError):
        Panel.from_arrays(fields, bad_time, symbols=("A",))


def test_panel_rejects_symbol_length() -> None:
    fields = {name: np.zeros((2, 4), dtype=np.float32) for name in PANEL_FIELDS}
    with pytest.raises(DataError):
        Panel.from_arrays(fields, np.arange(4, dtype=np.int64) * 3600, symbols=("A",))


def test_panel_missing_field() -> None:
    fields = {name: np.zeros((1, 4), dtype=np.float32) for name in PANEL_FIELDS if name != "volume"}
    with pytest.raises(DataError):
        Panel.from_arrays(fields, np.arange(4, dtype=np.int64) * 3600, symbols=("A",))


def test_panel_slice_view() -> None:
    p = _panel(2, 20)
    sub = p.slice_view(5, 12)
    assert sub.n_bars == 7
    assert np.array_equal(sub.time, p.time[5:12])
    assert np.array_equal(sub.close, p.close[:, 5:12])
    assert sub.fingerprint != p.fingerprint
    with pytest.raises(DataError):
        p.slice_view(10, 10)


def test_panel_to_raw_dict() -> None:
    p = _panel(1, 6)
    raw = p.to_raw_dict()
    assert set(raw) == set(PANEL_FIELDS) | {"time"}
    assert raw["close"].shape == (1, 6)


def test_panel_fingerprint_deterministic_and_sensitive() -> None:
    p = _panel(2, 12)
    assert p.fingerprint == p.fingerprint
    assert p.fingerprint.startswith("sha256:")

    # 数据改变 → 指纹改变
    fields = {name: p.fields[name].copy() for name in PANEL_FIELDS}
    fields["close"][0, 0] += np.float32(1.0)
    mutated = Panel.from_arrays(
        fields, p.time, symbols=p.symbols, timeframe=p.timeframe,
        market_profile_name=p.market_profile_name,
    )
    assert mutated.fingerprint != p.fingerprint

    # symbols 顺序敏感
    reordered = Panel.from_arrays(
        p.fields, p.time, symbols=tuple(reversed(p.symbols)), timeframe=p.timeframe
    )
    assert reordered.fingerprint != p.fingerprint
