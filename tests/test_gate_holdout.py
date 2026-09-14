# -*- coding: utf-8 -*-
# Copyright (c) 2026 Original Author
# Copyright (c) 2026 Your Company
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""hold-out 一次性封印闸单测（M12 / 验收 #5）。"""

from __future__ import annotations

import pytest

from miaosuan.data.split import HoldoutSealRegistry
from miaosuan.errors import HoldoutSealedError
from miaosuan.gate.holdout import HoldoutGate


def _gate() -> HoldoutGate:
    return HoldoutGate(HoldoutSealRegistry(HoldoutSealRegistry.IN_MEMORY))


def test_seal_then_second_use_raises() -> None:
    gate = _gate()
    fp = "sha256:abc"
    assert gate.is_sealed(fp) is False
    gate.seal(fp, meta={"n_bars": 1000})
    assert gate.is_sealed(fp) is True
    assert gate.sealed_count() == 1
    with pytest.raises(HoldoutSealedError):
        gate.seal(fp)


def test_allow_evaluation_seal_once() -> None:
    gate = _gate()
    fp = "sha256:def"
    assert gate.allow_evaluation(fp, meta={"n_bars": 500}) is True
    with pytest.raises(HoldoutSealedError):
        gate.allow_evaluation(fp)


def test_assert_untouched() -> None:
    gate = _gate()
    fp = "sha256:ghi"
    gate.assert_untouched(fp)  # 未封印 → 通过（不抛）
    gate.seal(fp)
    with pytest.raises(HoldoutSealedError):
        gate.assert_untouched(fp)


def test_sealed_count_and_list() -> None:
    gate = _gate()
    assert gate.sealed_count() == 0
    gate.seal("sha256:z")
    gate.seal("sha256:a")
    assert gate.sealed_count() == 2
    assert gate.list_seals() == ["sha256:a", "sha256:z"]
