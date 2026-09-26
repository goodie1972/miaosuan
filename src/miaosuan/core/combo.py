# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""多因子组合信号层（combine，T1）—— 纯 numpy，不依赖 ``ir`` / 不读 env / 不做 IO。

规格：``docs/combo-impl-spec.md`` §1.4 的组合管线，**顺序不可更改**::

    成员因子（已 z-score）→ 符号对齐（按 IC 统一方向）→ 加权和
    → 因果滚动重标定（roll_window，无前视）→ **单次** tanh → **单次** neutral_band

三条硬约束（来自 §1.4，均有测试盯守）：

1. **禁止仓位层平均**：各弱因子的小信号在**合并前**就会被自己的中性带清零，
   平均出来只剩噪声；必须在**因子层**加权求和，最后只过一次 ``tanh`` + 一次中性带。
2. **因果无前视**：加权和之后的重标定复用 :meth:`StackVM._normalize_output` ——
   与单因子完全同一套归一化（滚动 z-score、warm-up 置 0、clip ±3、样本不足退化
   expanding），不另写一份，避免两条口径日後漂移。
3. **禁止 GA 搜权重**：权重由调用方确定性给出（等权 / IC 加权），本层只做数值。

依赖方向：``core.combo`` 只依赖同包 ``core.signal`` 与 ``core.vm``，**不反向依赖**
``ir`` / ``search`` / ``report``（架构依赖铁律）。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .signal import compute_target_positions_stateless
from .vm import DEFAULT_ROLL_WINDOW, StackVM

__all__ = [
    "combo_to_position",
    "combine_factors",
    "sign_align",
]


def sign_align(
    factors: np.ndarray | Sequence[np.ndarray],
    ic: np.ndarray | Sequence[float],
) -> np.ndarray:
    """按 IC 把各成员因子**翻到同一方向**（符号对齐），返回堆叠数组 ``[M, N, T]``。

    单因子的符号是任意的（``f`` 与 ``-f`` 等价），直接加权会让方向相反的成员
    互相抵消；本函数统一到「IC ≥ 0 的方向」，之后加权和才是同向汇聚。

    判定：``ic < 0`` 翻转（× -1）；``ic >= 0`` 或 **IC 缺失/NaN 保持原方向**
    （NaN 没有方向证据，翻转等于凭空掷硬币）。

    Args:
        factors: 成员因子序列（每项 ``[N, T]``），或已堆叠的 ``[M, N, T]`` 数组。
        ic: 与 ``factors`` 等长的 IC 序列（调用方算好传入；本层不算统计量）。

    Returns:
        ``[M, N, T]`` 的 float32 堆叠数组（**不修改入参**，纯函数）。

    Raises:
        ValueError: ``factors`` 为空、或 ``ic`` 与因子数量不一致。
    """
    if isinstance(factors, np.ndarray):
        stack = np.asarray(factors, dtype=np.float32)
    else:
        if len(factors) == 0:
            raise ValueError("factors 不能为空")
        stack = np.stack([np.asarray(f, dtype=np.float32) for f in factors], axis=0)
    if stack.ndim < 1:
        raise ValueError(f"factors 维数非法：{stack.ndim}（至少 1 维，常规为 [M,N,T]）")

    ic_arr = np.asarray(ic, dtype=np.float64).reshape(-1)
    if ic_arr.size != stack.shape[0]:
        raise ValueError(
            f"ic 数量与因子数量不一致：{ic_arr.size} vs {stack.shape[0]}"
        )
    # IC<0 才翻转；ic==0 / NaN 一律保持原方向（无方向证据不等于反向）。
    signs = np.where(np.isfinite(ic_arr) & (ic_arr < 0.0), -1.0, 1.0)
    # signs 是 float64，先转回 float32 再乘，避免整条链路被 upcast（下游按 float32 走）。
    aligned = stack * signs.reshape((-1,) + (1,) * (stack.ndim - 1)).astype(np.float32)
    return np.asarray(aligned, dtype=np.float32)


def combine_factors(
    factors: Sequence[np.ndarray],
    weights: np.ndarray | Sequence[float],
    *,
    roll_window: int = DEFAULT_ROLL_WINDOW,
) -> np.ndarray:
    """因子层加权和 + **因果滚动重标定**，返回 ``[N, T]`` 组合因子。

    顺序即规格 §1.4：先加权求和，再做一次因果重标定 —— **不是**先各自归一再
    平方和。加权和的量纲随 ``Σw²`` 缩放，重标定把它拉回 z-score 尺度，
    下游 ``tanh`` 才有可比的饱和度。

    Args:
        factors: 已（按 IC）符号对齐的成员因子序列，每项 ``[N, T]``，形状必须一致。
        weights: 与 ``factors`` 等长的权重（等权传 ``[1/M]*M``；不强制归一——
            后面的重标定对整体缩放不敏感）。
        roll_window: 因果滚动重标定窗口（默认取 :data:`DEFAULT_ROLL_WINDOW` = 500，
            与 :class:`Semantics.roll_window` / 单因子栈同值）。

    Returns:
        ``[N, T]`` float32 组合因子（已重标定、clip ±3、warm-up 置 0）。

    Raises:
        ValueError: 序列为空、长度不匹配、形状不一致、权重非有限或全零。
    """
    if len(factors) == 0:
        raise ValueError("factors 不能为空")
    # 先校验形状再 stack —— 否则 np.stack 会抢先抛出无信息的英文错误，
    # 调用方拿不到「哪个成员、和谁不一致」这类可行动线索。
    arrays = [np.asarray(f, dtype=np.float32) for f in factors]
    reference = arrays[0].shape
    for i, arr in enumerate(arrays[1:], start=1):
        if arr.shape != reference:
            raise ValueError(
                f"factors[{i}] 形状 {arr.shape} 与 factors[0] {reference} 不一致"
            )
    stack = np.stack(arrays, axis=0)

    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.size != len(factors):
        raise ValueError(f"weights 数量与因子数量不一致：{w.size} vs {len(factors)}")
    if not bool(np.all(np.isfinite(w))):
        raise ValueError("weights 含非有限值（NaN/Inf）")
    if float(np.abs(w).sum()) == 0.0:
        raise ValueError("weights 全为 0 —— 组合恒为常数，无信号可言")
    if np.any(~np.isfinite(stack)):
        raise ValueError("factors 含非有限值（NaN/Inf）——成员因子必须可求值")

    agg = np.tensordot(w.astype(np.float32), stack, axes=(0, 0))  # [N, T]
    # 因果重标定：复用单因子栈的归一化，杜绝两套口径（滚动 z / warm-up / clip ±3）。
    return np.asarray(
        StackVM._normalize_output(np.asarray(agg, dtype=np.float32), int(roll_window)),
        dtype=np.float32,
    )


def combo_to_position(
    agg: np.ndarray,
    *,
    neutral_band: float,
    long_only: bool,
) -> np.ndarray:
    """组合因子 → 连续仓位：**单次** ``tanh`` + **单次** 中性带 + ``long_only`` 开关。

    直接复用 :func:`~miaosuan.core.signal.compute_target_positions_stateless`，
    与单因子路径共用同一段「tanh → 归零」语义（``long_only`` 与中性带都是逐点
    归零，先后顺序不改变结果），保证组合与单因子的仓位口径**逐点一致**。

    Args:
        agg: :func:`combine_factors` 的输出 ``[N, T]``（已因果重标定）。
        neutral_band: 中性带下界（``|pos| < band ⇒ 0``；默认 0.05 = ``MIN_TRADE_EXPOSURE``）。
        long_only: ``True`` 时负仓位归零（``long_short=False`` 的开关形式）。

    Returns:
        与 ``agg`` 同形状的连续仓位 ``[-1, +1]``。

    Raises:
        ValueError: ``neutral_band`` 为负或非有限。
    """
    band = float(neutral_band)
    if not np.isfinite(band) or band < 0.0:
        raise ValueError(f"neutral_band 非法：{neutral_band!r}（须为有限且 >= 0）")
    return compute_target_positions_stateless(
        np.asarray(agg, dtype=np.float32),
        min_trade_exposure=band,
        long_only=bool(long_only),
    )
