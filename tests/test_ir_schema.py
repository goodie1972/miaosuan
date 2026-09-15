# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""M13：``StrategySpec`` v1 结构、往返与指纹稳定性。"""

from __future__ import annotations

import dataclasses

import pytest

from miaosuan.ir.schema import (
    SPEC_VERSION,
    Evidence,
    FactorPayload,
    ParamPayload,
    Provenance,
    Semantics,
    StrategySpec,
)

_TOKENS: tuple[int, ...] = (33, 62, 3, 87, 72, 119, 73, 103)


def _spec(**kwargs: object) -> StrategySpec:
    base: dict[str, object] = {
        "name": "h1_xauusd_miaosuan",
        "payload": FactorPayload(tokens=_TOKENS, vocab_version="v9217a2c0d91a"),
        "semantics": Semantics(),
        "evidence": Evidence(n_trials=100, wf_folds=5, val_score=1.5, gate_verdict="DEPLOYABLE"),
    }
    base.update(kwargs)
    return StrategySpec(**base)  # type: ignore[arg-type]


def test_default_spec_version_is_v1() -> None:
    assert SPEC_VERSION == "1.0"
    assert StrategySpec().spec_version == SPEC_VERSION


def test_factor_roundtrip() -> None:
    spec = _spec()
    restored = StrategySpec.from_dict(spec.to_dict())
    assert restored == spec
    assert restored.tokens == _TOKENS


def test_param_payload_roundtrip() -> None:
    spec = _spec(payload=ParamPayload(template_id="tpl_v1", params={"n": 5.0, "m": 0.25}))
    restored = StrategySpec.from_dict(spec.to_dict())
    assert isinstance(restored.payload, ParamPayload)
    assert restored.payload.template_id == "tpl_v1"
    assert restored.payload.params["m"] == 0.25
    assert restored.tokens == ()


def test_unknown_payload_kind_raises() -> None:
    with pytest.raises(ValueError, match="未知 payload.kind"):
        StrategySpec.from_dict({"payload": {"kind": "mystery"}})


def test_spec_id_is_stable_across_timestamps() -> None:
    early = _spec(provenance=Provenance(created_at="2026-01-01T00:00:00+00:00", seed=1))
    late = _spec(provenance=Provenance(created_at="2026-09-10T12:00:00+00:00", seed=1))
    assert early.spec_id == late.spec_id


def test_spec_id_changes_with_tokens_and_semantics() -> None:
    a = _spec()
    b = _spec(payload=FactorPayload(tokens=(33, 62, 3, 87, 72, 119, 73, 104), vocab_version="v9217a2c0d91a"))
    c = _spec(semantics=Semantics(neutral_band=0.1))
    assert a.spec_id != b.spec_id
    assert a.spec_id != c.spec_id


def test_spec_id_ignores_evidence_and_git() -> None:
    a = _spec(evidence=Evidence(n_trials=1))
    b = _spec(evidence=Evidence(n_trials=999, gate_verdict="BLOCKED"))
    assert a.spec_id == b.spec_id


def test_evidence_deployable_property() -> None:
    assert Evidence(gate_verdict="DEPLOYABLE").deployable
    assert not Evidence(gate_verdict="RESEARCH_ONLY").deployable
    assert not Evidence(gate_verdict="BLOCKED").deployable


def test_semantics_defaults_match_core() -> None:
    sem = Semantics()
    assert sem.position_fn == "tanh"
    assert sem.neutral_band == 0.05
    assert sem.long_short is True
    assert sem.warmup_bars == 499
    assert sem.roll_window == 500


def test_from_dict_tolerates_missing_optional_sections() -> None:
    spec = StrategySpec.from_dict(
        {"payload": {"kind": "factor", "tokens": [1, 2], "vocab_version": "v1"}}
    )
    assert spec.payload.tokens == (1, 2)
    assert spec.evidence.gate_verdict == "UNKNOWN"
    assert spec.provenance.git_sha == "unknown"


def test_spec_is_frozen() -> None:
    spec = _spec()
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.name = "mutated"  # type: ignore[misc]
