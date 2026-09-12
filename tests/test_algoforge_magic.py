"""M15：magic 号段账本的分配、去重与幂等。"""

from __future__ import annotations

import json
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
    assert isinstance(magic, int), f"magic 必须是 int（平台侧 magic: int），实际 {type(magic)}"
    assert str(magic) not in KNOWN_MAGICS
    assert str(magic).startswith("66")
    assert len(str(magic)) == MAGIC_LENGTH
    assert int(str(magic)[2:4]) >= 18


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
    assert str(v1)[4:] == "01" and str(v2)[4:] == "02"


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
    assert {int(k) for k in KNOWN_MAGICS} <= used


def test_legacy_string_ledger_is_migrated_and_respected(tmp_path: Path) -> None:
    """老账本（magic 存成字符串）必须照常工作：不失效、不被重复分配。

    AlgoForge 侧 ``magic: int``，改成 int 后若不做兼容，已导出策略（老 magic 是
    字符串）会既查不到、又可能被重新分配给别的策略 —— 实盘事故级。
    """
    path = tmp_path / "magic.json"
    path.write_text(
        json.dumps({
            "version": 1,
            "records": [
                {"magic": "661801", "name": "h1_xauusd_miaosuan", "version": 1},
                {"magic": "661802", "name": "h1_btc_miaosuan", "version": 1},
            ],
        }),
        encoding="utf-8",
    )
    ledger = MagicLedger(path)

    # ① 读进来统一成 int
    assert ledger.find("h1_xauusd_miaosuan", 1) == 661801
    assert isinstance(ledger.find("h1_xauusd_miaosuan", 1), int)
    assert {661801, 661802} <= ledger.used

    # ② 幂等：同一 (name, version) 仍拿到老号，绝不重新分配
    assert ledger.allocate("h1_xauusd_miaosuan", 1) == 661801

    # ③ 新分配必须避开老号
    fresh = ledger.allocate("brand_new", 1)
    assert isinstance(fresh, int)
    assert fresh not in {661801, 661802}

    # ④ 落盘后已迁移成 int，老记录一条不少
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert all(isinstance(r["magic"], int) for r in saved["records"])
    assert {r["magic"] for r in saved["records"]} == {661801, 661802, fresh}


def test_magic_roundtrips_as_int_through_ledger(tmp_path: Path) -> None:
    """新账本里 magic 落盘即为 int，不再产生字符串。"""
    path = tmp_path / "magic.json"
    magic = allocate_magic("roundtrip", 1, ledger_path=path)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["records"][0]["magic"] == magic
    assert isinstance(saved["records"][0]["magic"], int)


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
