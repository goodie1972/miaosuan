"""M15：magic 号段账本的分配、去重与幂等。"""

from __future__ import annotations

from pathlib import Path

import pytest

from miaosuan.adapters.algoforge.contract import KNOWN_MAGICS, MAGIC_LENGTH, MAGIC_PREFIX
from miaosuan.adapters.algoforge.magic_registry import (
    MagicLedger,
    allocate_magic,
    describe,
    known_magics,
)


def test_known_magics_are_66_prefixed() -> None:
    for magic in KNOWN_MAGICS:
        assert magic.startswith(MAGIC_PREFIX)
        assert len(magic) == MAGIC_LENGTH
    assert "661601" in known_magics()
    assert "661701" in known_magics()


def test_allocate_avoids_known_magics(tmp_path: Path) -> None:
    ledger = MagicLedger(tmp_path / "magic.json")
    magic = ledger.allocate("brand_new", 1)
    assert magic not in KNOWN_MAGICS
    assert magic.startswith("66")
    assert len(magic) == MAGIC_LENGTH
    assert int(magic[2:4]) >= 18


def test_allocate_is_idempotent_for_same_name_and_version(tmp_path: Path) -> None:
    path = tmp_path / "magic.json"
    first = allocate_magic("xauusd_factor", 1, ledger_path=path)
    second = allocate_magic("xauusd_factor", 1, ledger_path=path)
    assert first == second
    assert len(MagicLedger(path).records) == 1


def test_allocate_distinguishes_versions(tmp_path: Path) -> None:
    path = tmp_path / "magic.json"
    v1 = allocate_magic("xauusd_factor", 1, ledger_path=path)
    v2 = allocate_magic("xauusd_factor", 2, ledger_path=path)
    assert v1 != v2
    assert v1[4:] == "01" and v2[4:] == "02"


def test_allocate_distinguishes_names(tmp_path: Path) -> None:
    path = tmp_path / "magic.json"
    a = allocate_magic("strategy_a", 1, ledger_path=path)
    b = allocate_magic("strategy_b", 1, ledger_path=path)
    assert a != b


def test_ledger_persists_and_reloads(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "magic.json"
    magic = allocate_magic("persisted", 1, ledger_path=path)
    assert path.exists()
    reloaded = MagicLedger(path)
    assert reloaded.find("persisted", 1) == magic
    assert magic in reloaded.used


def test_used_includes_builtin_magics(tmp_path: Path) -> None:
    used = MagicLedger(tmp_path / "magic.json").used
    assert KNOWN_MAGICS.keys() <= used


def test_corrupted_ledger_degrades_to_empty(tmp_path: Path) -> None:
    path = tmp_path / "magic.json"
    path.write_text("{not json", encoding="utf-8")
    assert MagicLedger(path).records == ()


def test_version_bounds(tmp_path: Path) -> None:
    ledger = MagicLedger(tmp_path / "magic.json")
    with pytest.raises(ValueError, match="version 必须落在"):
        ledger.allocate("bad", 0)
    with pytest.raises(ValueError, match="version 必须落在"):
        ledger.allocate("bad", 100)


def test_describe_builtin(tmp_path: Path) -> None:
    info = describe("661701", ledger_path=tmp_path / "magic.json")
    assert info["source"] == "builtin"
    assert "alphagate" in str(info["name"])


def test_describe_ledger_and_free(tmp_path: Path) -> None:
    path = tmp_path / "magic.json"
    magic = allocate_magic("described", 1, ledger_path=path)
    assert describe(magic, ledger_path=path)["source"] == "ledger"
    assert describe("669999", ledger_path=path)["source"] == "free"


def test_exhaustion_raises(tmp_path: Path) -> None:
    ledger = MagicLedger(tmp_path / "magic.json")
    for seq in range(18, 100):
        ledger._records.append({"magic": f"66{seq:02d}01", "name": f"s{seq}", "version": 1})
    with pytest.raises(ValueError, match="号段耗尽"):
        ledger.allocate("overflow", 1)
