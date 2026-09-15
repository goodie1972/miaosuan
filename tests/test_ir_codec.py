# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""M13：``StrategySpec`` 序列化的**确定性**与文件往返。"""

from __future__ import annotations

from pathlib import Path

import pytest

from miaosuan.ir.codec import (
    canonical_json,
    decode_spec,
    encode_spec,
    read_spec,
    write_spec,
)
from miaosuan.ir.schema import Evidence, FactorPayload, Provenance, StrategySpec


def _spec() -> StrategySpec:
    return StrategySpec(
        name="h1_xauusd_miaosuan",
        payload=FactorPayload(tokens=(33, 62, 3, 87), vocab_version="v9217a2c0d91a"),
        evidence=Evidence(
            n_trials=3280,
            wf_folds=5,
            val_score=2.327,
            cost_sensitivity={"0.5": 2.51, "2.0": 1.95},
            deflated_sharpe=0.87,
            gate_verdict="DEPLOYABLE",
            gate_reasons=("ok",),
        ),
        provenance=Provenance(git_sha="ce83c09", seed=42, market="FOREX_XAUUSD"),
    )


def test_encode_is_deterministic() -> None:
    assert encode_spec(_spec()) == encode_spec(_spec())


def test_encode_sorts_keys_at_every_level() -> None:
    text = encode_spec(_spec())
    assert text.index('"evidence"') < text.index('"name"')
    assert text.index('"payload"') < text.index('"provenance"')


def test_canonical_json_is_compact_and_sorted() -> None:
    assert canonical_json({"b": 1, "a": {"d": 2, "c": 3}}) == '{"a":{"c":3,"d":2},"b":1}'


def test_roundtrip_preserves_equality() -> None:
    spec = _spec()
    assert decode_spec(encode_spec(spec)) == spec


def test_roundtrip_preserves_unicode() -> None:
    spec = _spec()
    spec = StrategySpec(
        spec_version=spec.spec_version,
        name=spec.name,
        payload=spec.payload,
        semantics=spec.semantics,
        evidence=spec.evidence,
        provenance=spec.provenance,
        notes="中文备注：预热 499 根",
    )
    assert decode_spec(encode_spec(spec)).notes == "中文备注：预热 499 根"


def test_decode_rejects_non_object() -> None:
    with pytest.raises(ValueError, match="顶层必须是 JSON 对象"):
        decode_spec("[1, 2, 3]")


def test_write_and_read_roundtrip(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "spec.json"
    written = write_spec(target, _spec())
    assert written == target
    assert read_spec(target) == _spec()


def test_write_creates_parent_directories(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "c" / "spec.json"
    write_spec(target, _spec())
    assert target.exists()


def test_written_bytes_are_stable(tmp_path: Path) -> None:
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    write_spec(a, _spec())
    write_spec(b, _spec())
    assert a.read_bytes() == b.read_bytes()
