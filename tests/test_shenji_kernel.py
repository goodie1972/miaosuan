"""M15 支撑：内核提取器 —— 覆盖全部算子/特征，且不允许出现未定义符号。"""

from __future__ import annotations

import pytest

from miaosuan.adapters.shenji.kernel import (
    KernelExtractionError,
    build_kernel_plan,
)
from miaosuan.core.ops import OPS_CONFIG
from miaosuan.core.vocab import FORMULA_VOCAB

OFFSET: int = FORMULA_VOCAB.operator_offset
#: 与 AlphaMaster 对齐的最优公式（TRIX_15 → … → SCALE）。
AM_BEST: tuple[int, ...] = (33, 62, 3, 87, 72, 119, 73, 103)


def _formula_for_op(index: int, arity: int) -> tuple[int, ...]:
    """按 arity 拼一条最小可提取公式（不保证语义有效，只为触发提取）。"""
    if arity == 1:
        return (0, OFFSET + index)
    if arity == 2:
        return (0, 1, OFFSET + index)
    return (0, 1, 3, OFFSET + index)


def test_am_best_plan_shape() -> None:
    plan = build_kernel_plan(AM_BEST)
    assert plan.n_features == 3
    assert plan.n_ops == 5
    assert [name for _t, name, _f in plan.feature_entries] == [
        "TRIX_15",
        "RET_ENTROPY_20",
        "MA_DIFF",
    ]
    assert plan.op_names == ("MOMENTUM_5", "GATE", "SQRT", "JUMP", "SCALE")


def test_am_best_blocks_are_namespaced() -> None:
    plan = build_kernel_plan(AM_BEST)
    joined = "\n".join(plan.blocks)
    # 两个 core 模块都定义了 _EPS / _ema_simple，必须被前缀隔离
    assert "ft_EPS" in joined and "op_EPS" in joined
    assert "ft_ema_simple" not in joined or "op_ema_simple" not in joined


def test_blocks_exec_without_name_error() -> None:
    plan = build_kernel_plan(AM_BEST)
    namespace: dict[str, object] = {"np": __import__("numpy"), "Any": object, "math": __import__("math")}
    exec(  # noqa: S102 - 校验提取产物可执行，是保真度回归的前置条件
        compile("\n".join(plan.blocks), "<kernel>", "exec"), namespace
    )
    assert "ft_c_trix_15" in namespace  # 特征入口 wrapper（rename 保留 c 标记）
    assert "op_gate" in namespace


@pytest.mark.parametrize(
    "index", range(len(OPS_CONFIG)), ids=[f"op{OFFSET + i}" for i in range(len(OPS_CONFIG))]
)
def test_every_operator_extracts(index: int) -> None:
    name, _fn, arity = OPS_CONFIG[index]
    plan = build_kernel_plan(_formula_for_op(index, int(arity)))
    assert name in plan.op_names
    assert any(entry[0] == OFFSET + index for entry in plan.op_entries)


@pytest.mark.parametrize("feature_id", range(OFFSET), ids=lambda i: f"feat{i}")
def test_every_feature_extracts(feature_id: int) -> None:
    plan = build_kernel_plan((feature_id, OFFSET + 4))  # +4 = NEG（一元）
    assert [entry[0] for entry in plan.feature_entries] == [feature_id]


def test_unknown_symbol_would_raise() -> None:
    """闭包校验器本身可用：手工构造缺依赖的块必须报错。"""
    from miaosuan.adapters.shenji.kernel import _validate_blocks

    with pytest.raises(KernelExtractionError, match="未定义符号"):
        _validate_blocks(["def f(x):\n    return _totally_missing_helper(x)\n"])


def test_duplicate_tokens_do_not_duplicate_entries() -> None:
    plan = build_kernel_plan((0, 0, OFFSET + 4))
    assert [entry[0] for entry in plan.feature_entries] == [0, 0]
    # 特征函数只被解析一次（同名符号复用），块内不应重复定义
    joined = "\n".join(plan.blocks)
    assert joined.count("def ft_c_ret(") == 1
