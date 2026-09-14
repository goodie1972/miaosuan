# -*- coding: utf-8 -*-
# Copyright (c) 2026 Original Author
# Copyright (c) 2026 Your Company
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""RPN 表示/栈深/遗传算子单测（M10）。"""

from __future__ import annotations

import numpy as np
import pytest

from miaosuan.core.vocab import FORMULA_VOCAB
from miaosuan.search.rpn import (
    ARITY,
    FEATURE_COUNT,
    FORMULA_LEN,
    OPERATOR_OFFSET,
    VOCAB_SIZE,
    Individual,
    crossover,
    decode,
    depth_profile,
    hamming,
    is_feasible,
    mutate,
    random_feasible,
    repair,
    structure_violations,
)

#: AM 冻结最优公式（必须是可求值的 —— AM 约束采样器保证）。
AM_BEST = [33, 62, 3, 87, 72, 119, 73, 103]


def test_vocab_segmentation_frozen() -> None:
    assert FEATURE_COUNT == 65
    assert OPERATOR_OFFSET == 65
    assert VOCAB_SIZE == 127
    assert FORMULA_VOCAB.size == 127
    assert FORMULA_LEN == 8


def test_am_best_formula_is_feasible() -> None:
    assert is_feasible(AM_BEST)
    prof = depth_profile(AM_BEST)
    assert prof is not None
    assert prof[-1] == 1
    assert all(d >= 1 for d in prof), "任意前缀栈深必须 ≥ 1"


def test_all_features_sequence_infeasible() -> None:
    toks = list(range(8))  # 8 个特征 → 栈深 8 ≠ 1
    prof = depth_profile(toks)
    assert prof == [1, 2, 3, 4, 5, 6, 7, 8]
    assert not is_feasible(toks)


def test_operator_without_operands_infeasible() -> None:
    # 首个 token 就是二元算子 → 栈深不足 → 不可求值
    binary = next(t for t in ARITY if ARITY[t] == 2)
    assert depth_profile([binary, 0, 0]) is None
    assert not is_feasible([binary, 0, 0])


def test_decode_uses_token_names() -> None:
    text = decode(AM_BEST)
    names = FORMULA_VOCAB.token_names
    assert text.startswith(names[33])
    assert " → " in text
    assert len(text.split(" → ")) == 8


def test_hamming_basic() -> None:
    assert hamming([1, 2, 3], [1, 2, 3]) == 0
    assert hamming([1, 2, 3], [1, 9, 3]) == 1
    with pytest.raises(ValueError):
        hamming([1, 2], [1, 2, 3])


def test_random_feasible_always_feasible() -> None:
    rng = np.random.default_rng(0)
    for _ in range(300):
        toks = random_feasible(rng, FORMULA_LEN)
        assert toks.shape == (FORMULA_LEN,)
        assert toks.dtype == np.int64
        assert is_feasible(toks)
        assert int(toks.min()) >= 0
        assert int(toks.max()) < VOCAB_SIZE


def test_random_feasible_avoid_infection() -> None:
    rng = np.random.default_rng(1)
    feasible_and_clean = 0
    for _ in range(80):
        toks = random_feasible(rng, FORMULA_LEN, avoid_infection=True)
        assert is_feasible(toks)
        if not structure_violations(toks):
            feasible_and_clean += 1
    assert feasible_and_clean > 0


def test_repair_fixes_infeasible() -> None:
    rng = np.random.default_rng(2)
    bad = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int64)  # 全特征 → 栈深 8
    fixed = repair(bad, rng, length=FORMULA_LEN)
    assert is_feasible(fixed)


def test_crossover_children_feasible() -> None:
    rng = np.random.default_rng(3)
    a = random_feasible(rng, FORMULA_LEN)
    b = random_feasible(rng, FORMULA_LEN)
    for _ in range(100):
        child = crossover(a, b, rng, length=FORMULA_LEN)
        assert is_feasible(child)
    with pytest.raises(ValueError):
        crossover(np.array([0, 1]), np.array([0, 1, 2, 3]), rng)


def test_mutate_children_feasible() -> None:
    rng = np.random.default_rng(4)
    base = random_feasible(rng, FORMULA_LEN)
    for _ in range(200):
        child = mutate(base, rng, p_point=0.5, p_reset=0.2)
        assert is_feasible(child)
        assert child.shape == (FORMULA_LEN,)


def test_mutate_point_preserves_stack_depth() -> None:
    # 关闭 reset（p_reset=0）后，单点同类替换不改变 arity → 栈深不变。
    rng = np.random.default_rng(5)
    base = random_feasible(rng, FORMULA_LEN)
    base_profile = depth_profile(base)
    for _ in range(100):
        child = mutate(base, rng, p_point=1.0, p_reset=0.0, n_points=1)
        assert depth_profile(child) == base_profile


def test_individual_helpers() -> None:
    rng = np.random.default_rng(6)
    toks = random_feasible(rng, FORMULA_LEN)
    ind = Individual(tokens=toks, train_score=1.0, val_score=2.0, fitness=2.0, status="ok")
    assert ind.key() == tuple(int(t) for t in toks)
    assert ind.is_feasible()
    assert ind.is_evaluated()
    other = Individual(tokens=mutate(toks, rng, p_point=1.0, p_reset=0.0))
    assert ind.hamming(other) >= 1
    assert isinstance(ind.decoded, str)
    unevaluated = Individual(tokens=toks)
    assert not unevaluated.is_evaluated()
