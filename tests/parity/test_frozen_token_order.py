"""名称/顺序冻结锁回归（M3）：妙算注册表必须与 ``frozen_token_order.json`` 逐项一致。

这份 fixture 由 ``scripts/gen_frozen_fixture.py`` 从**冻结的 AlphaMaster**实时提取
（见 ``tests/fixtures/frozen_token_order.json``）。此后 M3/M4 的回归**只读 fixture、
不再依赖 AM 可导入** —— 这是后续 numpy 化移植的「名称/顺序锁」：

* 名称、顺序任一改变 -> ``VOCAB_VERSION`` 漂移 -> M3/M4 会静默把旧产物判为「不兼容」；
* 本测试在 CI 阶段就把这种漂移拦下。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from miaosuan.core.features import FEATURE_REGISTRY
from miaosuan.core.ops import OPERATOR_REGISTRY
from miaosuan.core.vocab import FORMULA_VOCAB, VOCAB_VERSION

pytestmark = pytest.mark.parity

_FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "frozen_token_order.json"


@pytest.fixture(scope="session")
def frozen() -> dict:
    if not _FIXTURE.is_file():
        pytest.skip(f"缺少冻结清单：{_FIXTURE}（先跑 scripts/gen_frozen_fixture.py）")
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


def test_counts_match_frozen(frozen: dict) -> None:
    assert frozen["feature_count"] == 65
    assert frozen["operator_count"] == 62
    assert frozen["vocab_size"] == 127


def test_vocab_version_matches_frozen(frozen: dict) -> None:
    assert frozen["vocab_version"] == VOCAB_VERSION
    assert FORMULA_VOCAB.version == frozen["vocab_version"]


def test_feature_order_and_category_match_frozen(frozen: dict) -> None:
    actual = [(s.name, s.category) for s in FEATURE_REGISTRY.feature_specs]
    expected = [(f["name"], f["category"]) for f in frozen["features"]]
    assert actual == expected, _first_diff(actual, expected)


def test_operator_order_and_arity_match_frozen(frozen: dict) -> None:
    actual = [(s.name, s.arity) for s in OPERATOR_REGISTRY.operator_specs]
    expected = [(o["name"], o["arity"]) for o in frozen["operators"]]
    assert actual == expected, _first_diff(actual, expected)


def test_token_names_match_frozen(frozen: dict) -> None:
    assert list(FORMULA_VOCAB.feature_names) == frozen["feature_names"]
    assert list(FORMULA_VOCAB.operator_names) == frozen["operator_names"]


def test_vocab_ids_are_contiguous_and_disjoint(frozen: dict) -> None:
    """token id 分段：feature [0,F-1]、operator [F,F+O-1] 严格不相交（R3.3）。"""
    v = FORMULA_VOCAB
    assert v.operator_offset == v.feature_count == 65
    assert v.size == v.feature_count + len(v.operator_names) == 127


def _first_diff(actual: list, expected: list) -> str:
    if len(actual) != len(expected):
        return f"长度不同：实际 {len(actual)} != 期望 {len(expected)}"
    for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
        if a != e:
            return f"首个差异在索引 {i}：实际 {a!r} != 期望 {e!r}"
    return "无差异"
