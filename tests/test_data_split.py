# -*- coding: utf-8 -*-
# Copyright (c) 2026 Original Author
# Copyright (c) 2026 Your Company
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""三段切分 + purge/embargo + hold-out 一次性封印单测（M7）。

（批量）验收：同数据 + 同 seed → 切分可复现（指纹一致）；同一 hold-out 指纹第二次使用 →
抛 :class:`HoldoutSealedError`。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from miaosuan.config import SplitConfig
from miaosuan.data.panel import PANEL_FIELDS, Panel
from miaosuan.data.split import (
    DataSplit,
    HoldoutSealRegistry,
    make_split,
    three_way_split,
)
from miaosuan.errors import DataError, HoldoutSealedError


def _panel(n: int = 1000, seed: int = 3) -> Panel:
    rng = np.random.default_rng(seed)
    t = n
    fields = {
        name: rng.normal(100, 1, (1, t)).astype(np.float32) for name in PANEL_FIELDS
    }
    time = (np.arange(t, dtype=np.int64) + 1) * 3600
    return Panel.from_arrays(
        fields, time, symbols=("XAUUSD",), timeframe="H1", market_profile_name="FOREX_XAUUSD"
    )


# ── three_way_split ────────────────────────────────────────────────────────


def test_three_way_split_shape_and_gaps() -> None:
    idx = three_way_split(1000, ratios=(0.6, 0.2, 0.2), purge_gap=20, embargo=4, seed=1)
    rs = idx.ranges()
    assert rs["train"] == (0, 600)
    assert rs["val"][0] == 600 + 24
    assert rs["holdout"][1] == 1000
    # 段间间隔 = purge + embargo
    assert rs["val"][0] - rs["train"][1] == 24
    assert rs["holdout"][0] - rs["val"][1] == 24
    assert idx.gap == 24


def test_three_way_split_disjoint_and_ordered() -> None:
    idx = three_way_split(500, ratios=(0.6, 0.2, 0.2), purge_gap=20, embargo=4)
    assert idx.train[-1] < idx.val[0] < idx.val[-1] < idx.holdout[0]
    all_idx = np.concatenate([idx.train, idx.val, idx.holdout])
    assert np.unique(all_idx).size == all_idx.size, "三段不得重叠"


def test_three_way_split_bad_ratios() -> None:
    with pytest.raises(DataError):
        three_way_split(1000, ratios=(0.5, 0.2, 0.2))


def test_three_way_split_too_short() -> None:
    with pytest.raises(DataError):
        three_way_split(10, ratios=(0.6, 0.2, 0.2), purge_gap=20, embargo=4)


# ── 可复现 ──────────────────────────────────────────────────────────────────


def test_split_reproducible_same_seed() -> None:
    panel = _panel(1200)
    cfg = SplitConfig()
    reg = HoldoutSealRegistry(HoldoutSealRegistry.IN_MEMORY)
    a = make_split(panel, cfg, registry=reg, seed=20260910, seal=False)
    b = make_split(panel, cfg, registry=reg, seed=20260910, seal=False)
    assert a.holdout_seal == b.holdout_seal
    assert isinstance(a, DataSplit)
    assert np.array_equal(a.indices.train, b.indices.train)  # type: ignore[union-attr]
    assert np.array_equal(a.indices.holdout, b.indices.holdout)  # type: ignore[union-attr]
    assert a.holdout.fingerprint == b.holdout.fingerprint
    assert not a.is_sealed()


def test_split_seed_changes_fingerprint() -> None:
    panel = _panel(1200)
    cfg = SplitConfig()
    reg = HoldoutSealRegistry(HoldoutSealRegistry.IN_MEMORY)
    a = make_split(panel, cfg, registry=reg, seed=1, seal=False)
    b = make_split(panel, cfg, registry=reg, seed=2, seal=False)
    assert a.holdout_seal != b.holdout_seal


# ── hold-out 一次性封印 ─────────────────────────────────────────────────────


def test_holdout_sealed_second_use_raises() -> None:
    panel = _panel(1200)
    cfg = SplitConfig()
    reg = HoldoutSealRegistry(HoldoutSealRegistry.IN_MEMORY)

    first = make_split(panel, cfg, registry=reg, seed=42, seal=True)
    assert first.is_sealed()
    assert reg.is_sealed(first.holdout_seal)

    with pytest.raises(HoldoutSealedError) as exc:
        make_split(panel, cfg, registry=reg, seed=42, seal=True)
    assert exc.value.code == "E-HOLDOUT-SEALED"


def test_sealed_different_data_allowed() -> None:
    cfg = SplitConfig()
    reg = HoldoutSealRegistry(HoldoutSealRegistry.IN_MEMORY)
    a = make_split(_panel(1000, seed=1), cfg, registry=reg, seed=7, seal=True)
    b = make_split(_panel(1000, seed=2), cfg, registry=reg, seed=7, seal=True)
    assert a.holdout_seal != b.holdout_seal
    assert len(reg.list_seals()) == 2


def test_registry_file_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "holdout_seals.json"
    reg = HoldoutSealRegistry(path)
    reg.seal("sha256:deadbeef", meta={"n_bars": 1000})
    assert path.is_file()

    reopened = HoldoutSealRegistry(path)
    assert reopened.is_sealed("sha256:deadbeef")
    assert reopened.list_seals() == ["sha256:deadbeef"]
    with pytest.raises(HoldoutSealedError):
        reopened.seal("sha256:deadbeef")


def test_registry_corrupt_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "holdout_seals.json"
    path.write_text("{not json", encoding="utf-8")
    reg = HoldoutSealRegistry(path)
    with pytest.raises(DataError):
        reg.is_sealed("x")
