# -*- coding: utf-8 -*-
"""策略说明文档生成器 + 映射表 单元测试。

覆盖场景：
1. ``write_strategy_docs`` 接受 **dict** 形式的 spec（模拟从 JSON 加载）
2. ``write_strategy_docs`` 接受真正的 ``StrategySpec`` 对象
3. 中英文 README 内容完整性（关键段落齐全）
4. ``spec_strategy_map.json`` 合并/去重逻辑
5. ``_coerce_spec`` 对非 dict 输入原样返回
6. ``_param_to_dict`` 对各种类型的容错
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from miaosuan.ir.schema import Evidence, StrategySpec, FactorPayload, Semantics, Provenance
from miaosuan.strategy_docs import (
    MAP_FILENAME,
    _coerce_spec,
    _param_to_dict,
    generate_readme_en,
    generate_readme_zh,
    update_strategy_map,
    write_strategy_docs,
)


# ── fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_spec_obj() -> StrategySpec:
    """构造一个完整的 StrategySpec 对象。"""
    return StrategySpec(
        name="test_strategy",
        payload=FactorPayload(tokens=(21, 25, 65, 69), vocab_version="v9217a2c0d91a"),
        semantics=Semantics(
            roll_window=500,
            warmup_bars=499,
            neutral_band=0.05,
            long_short=True,
            timeframe="M30",
        ),
        evidence=Evidence(
            n_trials=1,
            wf_folds=0,
            val_score=None,
            holdout_sharpe=None,
            cost_sensitivity={},
            deflated_sharpe=None,
            gate_verdict="RESEARCH_ONLY",
            gate_reasons=("reason A", "reason B"),
        ),
        provenance=Provenance(
            git_sha="abc123",
            vocab_version="v9217a2c0d91a",
            data_fingerprint="",
            seed=0,
            created_at="2026-10-01T00:00:00+00:00",
            miaosuan_version="0.1.0",
            market="FOREX_XAUUSD",
            budget="manual",
            config_snapshot={},
        ),
        notes="test",
    )


@pytest.fixture
def sample_spec_dict(sample_spec_obj: StrategySpec) -> dict:
    """从 StrategySpec 对象序列化为 dict（模拟 JSON 文件加载后的状态）。"""
    return sample_spec_obj.to_dict()


@pytest.fixture
def tmp_strategies(tmp_path: Path) -> Path:
    """创建一个临时 strategies 目录。"""
    d = tmp_path / "strategies"
    d.mkdir()
    return d


# ── 1. dict 输入 ──────────────────────────────────────────────────────────────


def test_write_strategy_docs_with_dict(sample_spec_dict: dict, tmp_strategies: Path):
    """write_strategy_docs 应接受 dict 形式的 spec 并正确产出文件。"""
    strategy_path = tmp_strategies / "test_v1.py"
    strategy_path.write_text("# placeholder", encoding="utf-8")

    written = write_strategy_docs(
        spec=sample_spec_dict,
        strategy_path=strategy_path,
        magic=661102,
        formula="-(BOLL_POS + MFI14)",
        param_space=[],
        spec_file="test_spec.json",
    )
    assert len(written) == 3
    zh, en, mp = written
    assert zh.name == "test_v1_README_zh.md"
    assert en.name == "test_v1_README_en.md"
    assert mp.name == MAP_FILENAME
    assert zh.is_file()
    assert en.is_file()
    assert mp.is_file()


# ── 2. StrategySpec 对象输入 ──────────────────────────────────────────────────


def test_write_strategy_docs_with_obj(sample_spec_obj: StrategySpec, tmp_strategies: Path):
    """write_strategy_docs 应接受真正的 StrategySpec 对象。"""
    strategy_path = tmp_strategies / "obj_v1.py"
    strategy_path.write_text("# placeholder", encoding="utf-8")

    written = write_strategy_docs(
        spec=sample_spec_obj,
        strategy_path=strategy_path,
        magic=42,
        formula="A + B",
        param_space=[],
        spec_file="obj_spec.json",
    )
    assert len(written) == 3
    zh_text = written[0].read_text(encoding="utf-8")
    assert "test_strategy" in zh_text


# ── 3. README 内容完整性 ─────────────────────────────────────────────────────


def test_readme_zh_sections(sample_spec_obj: StrategySpec):
    """中文 README 应包含所有六个段落。"""
    md = generate_readme_zh(
        sample_spec_obj,
        strategy_file="test_v1.py",
        magic=661102,
        formula="-(BOLL_POS + MFI14)",
        param_space=[],
    )
    for heading in ["一、公式", "二、语义", "三、风控", "四、可调参数", "五、证据", "六、溯源"]:
        assert heading in md, f"Missing section: {heading}"
    assert "RESEARCH_ONLY" in md
    assert "661102" in md


def test_readme_en_sections(sample_spec_obj: StrategySpec):
    """英文 README 应包含所有六个段落。"""
    md = generate_readme_en(
        sample_spec_obj,
        strategy_file="test_v1.py",
        magic=661102,
        formula="-(BOLL_POS + MFI14)",
        param_space=[],
    )
    for heading in ["1. Formula", "2. Semantics", "3. Risk control", "4. Tunable", "5. Evidence", "6. Provenance"]:
        assert heading in md, f"Missing section: {heading}"


# ── 4. 映射表合并/去重 ───────────────────────────────────────────────────────


def test_map_merge_dedup(tmp_strategies: Path):
    """update_strategy_map 应按 strategy_file 去重，重复导出更新而非追加。"""
    entry1 = {
        "strategy_file": "s1.py",
        "spec_id": "abc123",
        "name": "s1",
        "generated_at": "2026-10-01T00:00:00+00:00",
    }
    entry2 = {
        "strategy_file": "s1.py",  # same file → should overwrite
        "spec_id": "abc456",
        "name": "s1_updated",
        "generated_at": "2026-10-01T01:00:00+00:00",
    }
    update_strategy_map(tmp_strategies, entry1)
    update_strategy_map(tmp_strategies, entry2)

    map_path = tmp_strategies / MAP_FILENAME
    data = json.loads(map_path.read_text(encoding="utf-8"))
    assert len(data["strategies"]) == 1, "Duplicate not deduped!"
    assert data["strategies"][0]["spec_id"] == "abc456"


def test_map_multiple_entries(tmp_strategies: Path):
    """不同策略文件应追加为多条目。"""
    for i in range(3):
        update_strategy_map(
            tmp_strategies,
            {
                "strategy_file": f"s{i}.py",
                "spec_id": f"id{i}",
                "name": f"s{i}",
                "generated_at": f"2026-10-0{i+1}T00:00:00+00:00",
            },
        )
    map_path = tmp_strategies / MAP_FILENAME
    data = json.loads(map_path.read_text(encoding="utf-8"))
    assert len(data["strategies"]) == 3


# ── 5. _coerce_spec ──────────────────────────────────────────────────────────


def test_coerce_spec_with_dict(sample_spec_dict: dict):
    """_coerce_spec 应把 dict 转成 StrategySpec。"""
    result = _coerce_spec(sample_spec_dict)
    assert isinstance(result, StrategySpec)
    assert result.name == sample_spec_dict["name"]


def test_coerce_spec_with_obj(sample_spec_obj: StrategySpec):
    """_coerce_spec 对非 dict 输入应原样返回。"""
    result = _coerce_spec(sample_spec_obj)
    assert result is sample_spec_obj


# ── 6. _param_to_dict ─────────────────────────────────────────────────────────


def test_param_to_dict_with_dict():
    d = _param_to_dict({"name": "p1", "default": 1, "kind": "int", "description": "test"})
    assert d["name"] == "p1"


def test_param_to_dict_with_string():
    d = _param_to_dict("just_a_name")
    assert d["name"] == "just_a_name"


def test_param_to_dict_with_object():
    class FakeParam:
        def to_dict(self):
            return {"name": "obj_param", "default": 42, "kind": "float", "description": "obj"}

    d = _param_to_dict(FakeParam())
    assert d["name"] == "obj_param"


def test_param_to_dict_with_int():
    d = _param_to_dict(123)
    assert d["name"] == "123"
