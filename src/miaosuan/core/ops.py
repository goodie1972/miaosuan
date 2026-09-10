"""算子注册（M2 脚手架 —— 仅注册名称与 arity，**不**含 numpy 实现）。

.. warning::

   本文件在 T01/M2 阶段是**过渡脚手架**。它的唯一职责是：以与冻结 AM
   （``model_core/ops.py`` 的 ``_INITIAL_OPERATORS`` / ``_CROSS_SECTIONAL_OPERATORS`` /
   ``_TASK33_OPERATORS`` / ``_TASK34_OPERATORS`` 拼接后的顺序）**逐元素一致**的
   名称与顺序注册 **62** 个算子，从而让 :mod:`miaosuan.core.vocab` 能派生出于 AM
   **逐字符相同**的 ``VOCAB_VERSION``（== ``v9217a2c0d91a``）。

   **M3 将把本文件替换为 numpy 化实现**（填充 ``transform`` 的真实张量计算逻辑）。
   届时**必须保持**算子名称与顺序**一字不改**，否则 VOCAB_VERSION 会漂移。

算子分段（与 AM 一致，顺序不可变更）：

  * 0..43  初始 44 个算子；
  * 44..46 跨截面算子（Cross_Sectional，Task 3.2）；
  * 47..54 Task 3.3 追加 8 个；
  * 55..61 Task 3.4 追加 7 个。

注：架构文档 §1.3/§1.4 称「66 个算子」，经对冻结 AM 实测为 **62 个**；以实测为准
（62 才能派生出 ``v9217a2c0d91a``）。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .registry import OperatorSpec, Registry

__all__ = [
    "OPERATOR_REGISTRY",
    "OPERATOR_NAMES",
    "OPERATOR_COUNT",
    "OPS_CONFIG",
    "is_registered",
]


def _pending_operator(name: str, arity: int) -> Callable[..., Any]:
    """构造占位 transform 函数（M3 前调用即显式失败）。

    采用可变位置参数 ``*operands``，使注册层的 arity 观测校验跳过（与 AM 对
    二元/三元算子的 ``*operands`` 包装一致），仅以显式声明的 arity 为准。
    """

    def _transform(*operands: Any) -> Any:
        raise NotImplementedError(
            f"算子 '{name}'(arity={arity}) 的 numpy 实现将在 M3 提供；当前仅为词表脚手架。"
        )

    _transform.__name__ = f"_pending_{name}"
    return _transform


# ── 62 个算子的 (name, arity) ────────────────────────────────────────────
# 顺序即 token/算子维顺序，逐项对齐 AM model_core/ops.py。
_OPERATOR_DEFS: list[tuple[str, int]] = [
    # ── 初始 44 个算子 ───────────────────────────────────────────────
    # 基础算子 (0..11)
    ("ADD", 2),
    ("SUB", 2),
    ("MUL", 2),
    ("DIV", 2),
    ("NEG", 1),
    ("ABS", 1),
    ("SIGN", 1),
    ("GATE", 3),
    ("JUMP", 1),
    ("DECAY", 1),
    ("DELAY1", 1),
    ("MAX3", 1),
    # 时序算子 (12..21)
    ("TS_MEAN_5", 1),
    ("TS_MEAN_10", 1),
    ("TS_MEAN_20", 1),
    ("TS_STD_5", 1),
    ("TS_STD_10", 1),
    ("TS_STD_20", 1),
    ("TS_RANK_5", 1),
    ("TS_RANK_10", 1),
    ("TS_RANK_20", 1),
    ("TS_CORR_10", 2),
    # 趋势 / 动量类算子 (22..27)
    ("MOMENTUM_5", 1),
    ("MOMENTUM_10", 1),
    ("TS_MAX_10", 1),
    ("TS_MIN_10", 1),
    ("WMA", 1),
    ("DELAY4", 1),
    # v3.0 新增算子 (28..33)
    ("EMA_5", 1),
    ("EMA_20", 1),
    ("TS_QUANTILE_10", 1),
    ("TS_SKEW_10", 1),
    ("TS_MIN_20", 1),
    ("TS_MAX_20", 1),
    # v3.0 Alpha 101 + 补充算子 (34..43)
    ("DELTA", 1),
    ("TS_ARG_MAX_5", 1),
    ("TS_ARG_MIN_5", 1),
    ("DECAY_LINEAR_5", 1),
    ("SCALE", 1),
    ("COVARIANCE_10", 2),
    ("PRODUCT_5", 1),
    ("SIGNED_POWER_2", 1),
    ("TS_DECAY_EXP_5", 1),
    ("DELTA_5", 1),
    # ── Task 3.2 跨截面算子 (44..46) ─────────────────────────────────
    ("CS_RANK", 1),
    ("CS_SCALE", 1),
    ("CS_NEUTRALIZE", 1),
    # ── Task 3.3 追加 8 个 (47..54) ──────────────────────────────────
    ("TS_SUM_5", 1),
    ("TS_SUM_10", 1),
    ("TS_SUM_20", 1),
    ("MIN", 2),
    ("MAX", 2),
    ("POWER", 1),
    ("SIGNED_LOG", 1),
    ("SQRT", 1),
    # ── Task 3.4 追加 7 个 (55..61) ──────────────────────────────────
    ("TS_ZSCORE_10", 1),
    ("TS_ZSCORE_20", 1),
    ("WINSORIZE", 1),
    ("CLIP", 1),
    ("SIGMOID", 1),
    ("TANH_SQUASH", 1),
    ("IF_GT", 3),
]


# ── 构建 OPERATOR_REGISTRY 并派生 OPS_CONFIG 导出视图 ──────────────────────

OPERATOR_REGISTRY = Registry()

for _name, _arity in _OPERATOR_DEFS:
    OPERATOR_REGISTRY.register_operator(
        OperatorSpec(name=_name, arity=_arity, transform=_pending_operator(_name, _arity))
    )

# 由注册表导出有序算子名视图（保持下游 import 兼容）
OPERATOR_NAMES: tuple[str, ...] = OPERATOR_REGISTRY.operator_names

# 算子总数（= 62，与冻结 AM 一致）
OPERATOR_COUNT: int = len(OPERATOR_REGISTRY.operator_names)

# 导出视图：[(name, transform, arity), ...]，保持 AM 既有元组结构（下游兼容）
OPS_CONFIG: list[tuple[str, Callable[..., Any], int]] = [
    (spec.name, spec.transform, spec.arity) for spec in OPERATOR_REGISTRY.operator_specs
]


def is_registered(name: str) -> bool:
    """该 token 名称是否已注册（跨 Feature/Operator 全局唯一）。"""
    return name in OPERATOR_REGISTRY
