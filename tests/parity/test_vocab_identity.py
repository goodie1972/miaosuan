# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""★ 词表恒等回归验证：确保 VOCAB_VERSION 的确定性派生不变。

这是架构 §6.3 所称「最优雅的一条回归验证」：``VOCAB_VERSION`` 由
``sha256("\\n".join(token_names))[:12]`` 确定性派生。只要 token 组成与顺序不变，
版本字符串**必然相同**（当前产物为 ``v9217a2c0d91a``）。若实现中不小心改了顺序或
漏了算子，测试会立刻变红。

覆盖：
  * 与**冻结快照**逐字符比对（始终运行，离线，零外部依赖）；
  * 派生函数的性质（确定性、顺序敏感、成员敏感、无拼接歧义）。
"""

from __future__ import annotations

import pytest

from miaosuan.core import vocab as ms_vocab
from miaosuan.errors import VocabVersionMismatchError

pytestmark = pytest.mark.regression

# 冻结的期望版本（当前产物；同时见 strategies/best_XAUUSD.json）
FROZEN_VERSION = "v9217a2c0d91a"


# ── 1. 与冻结快照比对 ──────────────────────────────────────────────────────


def test_vocab_version_equals_frozen_snapshot(frozen_snapshot: dict) -> None:
    """妙算 VOCAB_VERSION 与冻结快照逐字符一致。"""
    assert frozen_snapshot["vocab_version"] == ms_vocab.VOCAB_VERSION
    assert ms_vocab.VOCAB_VERSION == FROZEN_VERSION


def test_schema_tag_equals_frozen_snapshot(frozen_snapshot: dict) -> None:
    assert frozen_snapshot["vocab_schema_tag"] == ms_vocab.VOCAB_SCHEMA_TAG


def test_feature_names_match_frozen_snapshot(frozen_snapshot: dict) -> None:
    """65 个特征名逐元素、逐顺序一致。"""
    expected = tuple(frozen_snapshot["feature_names"])
    actual = ms_vocab.FORMULA_VOCAB.feature_names
    assert actual == expected, _first_diff(actual, expected)


def test_operator_names_match_frozen_snapshot(frozen_snapshot: dict) -> None:
    """62 个算子名逐元素、逐顺序一致。"""
    expected = tuple(frozen_snapshot["operator_names"])
    actual = ms_vocab.FORMULA_VOCAB.operator_names
    assert actual == expected, _first_diff(actual, expected)


def test_counts_match_frozen_snapshot(frozen_snapshot: dict) -> None:
    """特征数 65 / 算子数 62 / 词表大小 127。"""
    v = ms_vocab.FORMULA_VOCAB
    assert v.feature_count == frozen_snapshot["feature_count"] == 65
    assert len(v.operator_names) == frozen_snapshot["operator_count"] == 62
    assert v.size == frozen_snapshot["vocab_size"] == 127
    assert v.operator_offset == v.feature_count


def test_snapshot_version_is_self_consistent(frozen_snapshot: dict) -> None:
    """快照自身自洽：其 token 名称重新派生的版本 == 记录的版本。"""
    token_names = tuple(frozen_snapshot["feature_names"]) + tuple(frozen_snapshot["operator_names"])
    assert ms_vocab.compute_vocab_version(token_names) == frozen_snapshot["vocab_version"]


# ── 2. 派生函数的性质 ─────────────────────────────────────────────────────


def test_compute_vocab_version_is_deterministic() -> None:
    names = ("A", "B", "C")
    assert ms_vocab.compute_vocab_version(names) == ms_vocab.compute_vocab_version(names)


def test_compute_vocab_version_order_sensitive() -> None:
    """顺序变化必改变版本（换行分隔，避免拼接歧义）。"""
    assert ms_vocab.compute_vocab_version(("A", "B")) != ms_vocab.compute_vocab_version(("B", "A"))


def test_compute_vocab_version_membership_sensitive() -> None:
    assert ms_vocab.compute_vocab_version(("A", "B")) != ms_vocab.compute_vocab_version(
        ("A", "B", "C")
    )


def test_compute_vocab_version_no_concat_ambiguity() -> None:
    """用换行作分隔符，("AB",) 与 ("A","B") 不得同版本。"""
    assert ms_vocab.compute_vocab_version(("AB",)) != ms_vocab.compute_vocab_version(("A", "B"))


def test_formula_vocab_version_property_uses_token_names() -> None:
    v = ms_vocab.FORMULA_VOCAB
    assert v.version == ms_vocab.compute_vocab_version(v.token_names)
    assert v.version == ms_vocab.VOCAB_VERSION


# ── 3. verify() 行为 ──────────────────────────────────────────────────────


def test_verify_accepts_matching_version() -> None:
    # 不抛异常即通过
    ms_vocab.FORMULA_VOCAB.verify(ms_vocab.VOCAB_VERSION)


def test_verify_rejects_mismatch() -> None:
    with pytest.raises(VocabVersionMismatchError) as excinfo:
        ms_vocab.FORMULA_VOCAB.verify("v000000000000")
    assert excinfo.value.code == "E-VOCAB-MISMATCH"


# ── 工具 ───────────────────────────────────────────────────────────────────


def _first_diff(actual: tuple, expected: tuple) -> str:
    """返回首个差异的可读描述（用于断言消息）。"""
    if len(actual) != len(expected):
        return f"长度不同：实际 {len(actual)} != 期望 {len(expected)}"
    for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
        if a != e:
            return f"首个差异在索引 {i}：实际 {a!r} != 期望 {e!r}"
    return "无差异"
