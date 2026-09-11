"""M13：可追溯信封构造与 git sha 采集。"""

from __future__ import annotations

import re
from pathlib import Path

from miaosuan.ir.provenance import (
    MIAOSUAN_VERSION,
    build_provenance,
    resolve_git_sha,
    short_sha,
    utc_now_iso,
)
from miaosuan.ir.schema import Provenance


def test_build_provenance_fills_envelope() -> None:
    prov = build_provenance(
        vocab_version="v9217a2c0d91a",
        data_fingerprint="deadbeef",
        seed=42,
        market="FOREX_XAUUSD",
        budget="standard",
        git_sha="ce83c09",
        created_at="2026-09-10T00:00:00+00:00",
        config_snapshot={"seed": 42},
    )
    assert prov.vocab_version == "v9217a2c0d91a"
    assert prov.data_fingerprint == "deadbeef"
    assert prov.seed == 42
    assert prov.market == "FOREX_XAUUSD"
    assert prov.budget == "standard"
    assert prov.git_sha == "ce83c09"
    assert prov.created_at == "2026-09-10T00:00:00+00:00"
    assert prov.miaosuan_version == MIAOSUAN_VERSION
    assert prov.config_snapshot == {"seed": 42}


def test_build_provenance_defaults_created_at_to_now() -> None:
    prov = build_provenance(vocab_version="v1")
    assert prov.created_at.endswith("+00:00")


def test_utc_now_iso_format() -> None:
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00$", utc_now_iso())


def test_short_sha() -> None:
    assert short_sha("abcdef1234567890") == "abcdef123456"
    assert short_sha("abcdef1234567890", length=7) == "abcdef1"
    assert short_sha("unknown") == "unknown"
    assert short_sha("") == ""


def test_resolve_git_sha_never_raises() -> None:
    sha = resolve_git_sha()
    assert sha == "unknown" or re.fullmatch(r"[0-9a-f]{40}", sha) is not None


def test_resolve_git_sha_returns_unknown_outside_repo(tmp_path: Path) -> None:
    # tmp_path 位于系统临时目录，通常不在 git 工作树内；即便命中父仓库也只可能是合法 sha
    sha = resolve_git_sha(tmp_path)
    assert sha == "unknown" or re.fullmatch(r"[0-9a-f]{40}", sha) is not None


def test_provenance_roundtrip() -> None:
    prov = build_provenance(vocab_version="v1", seed=7, market="M")
    assert Provenance.from_dict(prov.to_dict()) == prov
