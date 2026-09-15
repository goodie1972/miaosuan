# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""算子库（numpy 化移植，M3）。

本模块把 ``AlphaMaster`` 的 ``model_core/ops.py``（原 torch 实现，62 个算子）逐算子
移植为**纯 numpy** 实现，注册进 :data:`OPERATOR_REGISTRY`。``OPS_CONFIG`` 作为
「导出视图」由注册表派生，保持对下游 ``vocab.py`` / ``vm.py`` 的 import 兼容。

移植铁律（架构 §1.3、M3 任务书）：

* **名称与顺序一字不改**（44 初始 + 3 跨截面 + 8 Task3.3 + 7 Task3.4 = 62），否则
  ``VOCAB_VERSION`` 会漂移、旧产物被 ``verify()`` 拒绝；由
  ``tests/fixtures/frozen_token_order.json`` 与 ``tests/parity`` 回归锁定。
* **数值细节逐一对齐**，不做「顺手优化」：dtype 保持、``clamp`` 边界、``nan``/``inf``
  处理、除零保护 ``_EPS``、``nan_to_num`` 的默认 ``posinf``/``neginf`` 语义均与 AM 一致。
* **禁止** import torch、禁止文件 IO、禁止读取环境变量（CI 的
  ``tests/test_dependency_direction.py`` 强制校验）。

统一契约（沿用 AM R2.8, R2.9, R2.13）：

  - 形状契约：所有算子输入 ``[N, T]``、输出 ``[N, T]``（N=截面/样本，T=时间）。
  - 二元/三元算子在入口校验各操作数形状一致，不一致抛 :class:`ShapeError` 且不产出数组。
  - 时序算子一律**因果**（左侧零填充），不使用未来信息。

torch → numpy 关键等价：

  ``torch.unfold(1, d, 1)``        → ``np.lib.stride_tricks.sliding_window_view(xp, d, axis=1)``
  ``torch.clamp(v, lo, hi)``       → ``np.clip(v, lo, hi)``（NaN 均传播）
  ``torch.nan_to_num(x, nan=..)``  → ``_n2n(x, nan=..)``（posinf/neginf 默认同为 dtype 极值）
  ``(c > 0).float()``              → ``(c > 0).astype(np.float32)``
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .registry import OperatorSpec, Registry

__all__ = [
    "ShapeError",
    "OPERATOR_REGISTRY",
    "OPERATOR_NAMES",
    "OPERATOR_COUNT",
    "OPS_CONFIG",
    "is_registered",
]

# 除零保护常数（与 AM 一致）
_EPS = 1e-6


def _n2n(
    x: np.ndarray,
    nan: float = 0.0,
    posinf: float | None = None,
    neginf: float | None = None,
) -> np.ndarray:
    """``np.nan_to_num`` 的强类型包装。

    语义与 AM 的 ``torch.nan_to_num`` 一致：``nan`` 默认 0；``posinf``/``neginf``
    为 ``None`` 时替换为 dtype 的最大/最小有限值（numpy 与 torch 默认行为相同）。
    此处仅加类型注记（numpy 存根对 ``nan_to_num`` 的返回类型为 ``Any``）。
    """
    out: np.ndarray = np.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)
    return out


# ── 算子层错误类型（对应 design「错误类型模型」，归为算子层）──────────────


class ShapeError(Exception):
    """算子操作数形状不兼容或错误（R2.13）。

    二元/三元算子在入口发现各操作数形状不一致时抛出，且不产出数组。
    """


# ── 基础 / 时序 helper（逐行对齐 AM model_core/ops.py）────────────────────


def _ts_delay(x: np.ndarray, d: int) -> np.ndarray:
    """因果延迟 d 期（左侧零填充）。与 AM 的 zero-pad + 切片实现等价。"""
    if d == 0:
        return x
    out = np.zeros_like(x)
    out[:, d:] = x[:, :-d]
    return out


def _op_sign(x: np.ndarray) -> np.ndarray:
    """符号函数，**等价 torch.sign 的特殊值语义**。

    实测 torch 2.14 的 ``torch.sign(NaN) == 0.0``（``torch.sgn(NaN)`` 亦然），
    而 ``np.sign(NaN) == NaN``。为与 AM 逐元素对齐，这里用
    ``(x > 0) - (x < 0)``（NaN 的两侧比较均为 False -> 0），并保持输入 dtype。
    """
    out: np.ndarray = (x > 0).astype(x.dtype) - (x < 0).astype(x.dtype)
    return out


def _op_gate(condition: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """三路门控：condition>0 取 x，否则取 y。"""
    mask = (condition > 0).astype(np.float32)  # AM: (condition > 0).float()
    out: np.ndarray = mask * x + (1.0 - mask) * y
    return out


def _op_jump(x: np.ndarray) -> np.ndarray:
    """降低稀疏度：expanding 因果 zscore + tanh(z-1.5)，warm-up 前 5 根输出 0。

    因果 expanding zscore：每个 t 仅用 x[:, :t+1] 计算 mean/std，避免 look-ahead。
    warm-up（t < 5）输出 0（中性），避免 t=0 的常数输出污染因子起点。
    """
    _n, t = x.shape
    cnt = np.arange(1, t + 1, dtype=x.dtype).reshape(1, t)
    cumsum = np.cumsum(x, axis=1)
    mean = cumsum / cnt  # [N,T]，t 位 = x[:, :t+1].mean()
    cumsum_sq = np.cumsum(x * x, axis=1)
    var = (cumsum_sq / cnt) - mean * mean  # E[x^2] - E[x]^2
    std = np.sqrt(np.clip(var, 1e-12, None)) + _EPS
    z = (x - mean) / std
    out: np.ndarray = np.tanh(z - 1.5)  # tanh 软化，不再产生全零区间
    min_warmup = 5
    if t > min_warmup:
        out[:, :min_warmup] = 0.0
    return out


def _op_decay(x: np.ndarray) -> np.ndarray:
    """指数衰减加权：x + 0.8*x[t-1] + 0.6*x[t-2]，系数和 2.4 归一化。"""
    return (x + 0.8 * _ts_delay(x, 1) + 0.6 * _ts_delay(x, 2)) / 2.4


def _op_wma(x: np.ndarray) -> np.ndarray:
    """加权移动平均（权重 3,2,1），平滑信号。"""
    return (3.0 * x + 2.0 * _ts_delay(x, 1) + 1.0 * _ts_delay(x, 2)) / 6.0


# ── 时序滑动窗口辅助函数（因果，左侧零填充）───────────────────────────────


def _ts_rolling(x: np.ndarray, d: int) -> np.ndarray:
    """unfold 等价的因果滑动窗口，返回 ``[N, T, d]``。

    等价于 ``cat([zeros(N, d-1), x], dim=1).unfold(1, d, 1)``：
    左侧补 ``d-1`` 个 0，再滑窗，窗口数恰为 T。
    """
    n, _t = x.shape
    pad = np.zeros((n, d - 1), dtype=x.dtype)
    xp = np.concatenate([pad, x], axis=1)
    return np.lib.stride_tricks.sliding_window_view(xp, d, axis=1)  # [N, T, d]


def _ts_mean(x: np.ndarray, d: int) -> np.ndarray:
    """因果滑动均值，返回 ``[N, T]``。"""
    return _ts_rolling(x, d).mean(axis=-1)


def _ts_std(x: np.ndarray, d: int) -> np.ndarray:
    """因果滑动标准差（ddof=0），返回 ``[N, T]``，下界 1e-6。"""
    w = _ts_rolling(x, d)
    m = w.mean(axis=-1, keepdims=True)
    std = np.sqrt(((w - m) ** 2).mean(axis=-1)) + _EPS
    return _n2n(std, nan=0.0)


def _ts_rank(x: np.ndarray, d: int) -> np.ndarray:
    """因果滑动排名（严格小于当前值的比例），返回 ``[N, T]``，值域 ``[0, 1)``。"""
    w = _ts_rolling(x, d)
    cur = w[:, :, -1:]
    rank = (w < cur).astype(x.dtype).mean(axis=-1)
    return _n2n(rank, nan=0.0)


def _ts_corr_10(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """x 与 y 的 10 周期因果滑动 Pearson 相关系数，值域 [-1, 1]。

    窗口内 x 或 y 为常数（std < 1e-6）时该位置输出 0。
    """
    d = 10
    wx = _ts_rolling(x, d)
    wy = _ts_rolling(y, d)
    mx = wx.mean(axis=-1, keepdims=True)
    my = wy.mean(axis=-1, keepdims=True)
    cov = ((wx - mx) * (wy - my)).mean(axis=-1)
    sx = np.sqrt(((wx - mx) ** 2).mean(axis=-1))
    sy = np.sqrt(((wy - my) ** 2).mean(axis=-1))
    mask = (sx < 1e-6) | (sy < 1e-6)
    corr = cov / (sx * sy + 1e-8)
    corr = np.where(mask, 0.0, corr)
    return _n2n(corr, nan=0.0, posinf=0.0, neginf=0.0)


def _ema_simple(x: np.ndarray, span: int, exact: bool = False) -> np.ndarray:
    """指数加权移动平均（因果），span 期，exact 递推路径。

    统一使用 exact 递推，避免 T 大/小的路径分叉导致 train-serve skew。
    ``exact`` 为兼容参数，无论取值均走递推。
    """
    alpha = 2.0 / (span + 1.0)
    _n, t = x.shape
    if t == 0:
        return x.copy()
    if alpha >= 1.0:
        return x.copy()
    out = np.zeros_like(x)
    out[:, 0] = x[:, 0]
    for i in range(1, t):
        out[:, i] = alpha * x[:, i] + (1 - alpha) * out[:, i - 1]
    return out


def _ts_quantile(x: np.ndarray, d: int) -> np.ndarray:
    """当前值在过去 d 期的分位数（0~1），严格小于当前值，与 TS_RANK 语义统一。"""
    w = _ts_rolling(x, d)
    cur = w[:, :, -1:]
    rank = (w < cur).astype(x.dtype).mean(axis=-1)
    return _n2n(rank, nan=0.5)


def _ts_skew(x: np.ndarray, d: int) -> np.ndarray:
    """d 期偏度（三阶矩标准化），捕捉分布非对称性。"""
    w = _ts_rolling(x, d)
    m = w.mean(axis=-1, keepdims=True)
    s = np.sqrt(((w - m) ** 2).mean(axis=-1)) + _EPS
    skew = ((w - m) ** 3).mean(axis=-1) / (s**3)
    return _n2n(skew, nan=0.0, posinf=0.0, neginf=0.0)


def _delta(x: np.ndarray, d: int = 1) -> np.ndarray:
    """d 期差分：x[t] - x[t-d]，前 d 位置 0（Alpha101 最常用算子）。"""
    if d == 0:
        return x
    out = np.zeros_like(x)
    out[:, d:] = x[:, d:] - x[:, :-d]
    return out


def _ts_arg_max(x: np.ndarray, d: int) -> np.ndarray:
    """过去 d 期最大值位置（归一化到 [0,1]，0=最早，1=最近）。"""
    w = _ts_rolling(x, d)
    idx = w.argmax(axis=-1).astype(x.dtype)
    return idx / max(d - 1, 1)


def _ts_arg_min(x: np.ndarray, d: int) -> np.ndarray:
    """过去 d 期最小值位置（归一化到 [0,1]）。"""
    w = _ts_rolling(x, d)
    idx = w.argmin(axis=-1).astype(x.dtype)
    return idx / max(d - 1, 1)


def _decay_linear(x: np.ndarray, d: int) -> np.ndarray:
    """线性衰减加权平均（近期权重更高），权重 = [1..d]/sum。"""
    w = _ts_rolling(x, d)
    weights = np.arange(1, d + 1, dtype=x.dtype)
    weights = weights / weights.sum()
    out: np.ndarray = (w * weights).sum(axis=-1)
    return out


def _decay_exp(x: np.ndarray, d: int, alpha: float = 0.5) -> np.ndarray:
    """指数衰减加权平均（近期权重更高），权重 = alpha*(1-alpha)^i 归一化。"""
    w = _ts_rolling(x, d)
    weights = np.array([alpha * (1 - alpha) ** i for i in range(d)], dtype=x.dtype)
    weights = weights / weights.sum()
    out: np.ndarray = (w * weights).sum(axis=-1)
    return out


def _scale(x: np.ndarray) -> np.ndarray:
    """沿时间轴缩放到单位 L1 范数（因果累积和）：x[t] / sum(|x[..t]|)。"""
    cumsum = np.cumsum(np.abs(x), axis=1) + _EPS
    return x / cumsum


def _ts_covariance(x: np.ndarray, y: np.ndarray, d: int) -> np.ndarray:
    """d 期因果滑动协方差。"""
    wx = _ts_rolling(x, d)
    wy = _ts_rolling(y, d)
    mx = wx.mean(axis=-1, keepdims=True)
    my = wy.mean(axis=-1, keepdims=True)
    cov = ((wx - mx) * (wy - my)).mean(axis=-1)
    return _n2n(cov, nan=0.0)


def _ts_product(x: np.ndarray, d: int) -> np.ndarray:
    """d 期因果滑动乘积（对数域累加防爆炸）：prod = exp(sum(log1p(x)))。

    输入 clamp 到 [-0.999, +inf) 避免 log1p 产生 NaN；对数累加和 clamp 到 [-10,10]
    防止 expm1 溢出。
    """
    x_safe = np.clip(x, -0.999, None)
    log_x = np.log1p(x_safe)
    w = _ts_rolling(log_x, d)
    log_sum = np.clip(w.sum(axis=-1), -10.0, 10.0)
    out = np.expm1(log_sum)
    return _n2n(out, nan=0.0, posinf=0.0, neginf=0.0)


def _signed_power(x: np.ndarray, a: float = 2.0) -> np.ndarray:
    """带符号乘方：sign(x) * |x|^a，出口 clamp(-1e9, 1e9) 防溢出。"""
    out = np.sign(x) * np.abs(x) ** a
    return _n2n(np.clip(out, -1e9, 1e9), nan=0.0, posinf=0.0, neginf=0.0)


def _power_signed(x: np.ndarray, a: float = 2.0) -> np.ndarray:
    """符号幂变换（R2.5）：sign(x)*|x|^a，出口 clamp(-1e9, 1e9)。"""
    return _signed_power(x, a)


def _signed_log(x: np.ndarray) -> np.ndarray:
    """带符号自然对数：sign(x)*log1p(|x|)，全实数域安全。"""
    out = np.sign(x) * np.log1p(np.abs(x))
    return _n2n(out, nan=0.0, posinf=0.0, neginf=0.0)


def _signed_sqrt(x: np.ndarray) -> np.ndarray:
    """带符号平方根：sign(x)*sqrt(|x|)，负输入不产 NaN。"""
    out = np.sign(x) * np.sqrt(np.abs(x))
    return _n2n(out, nan=0.0, posinf=0.0, neginf=0.0)


def _ts_sum(x: np.ndarray, d: int) -> np.ndarray:
    """因果滚动求和（R2.3）：左侧零填充 + 滑窗，每步只用 [t-w+1..t]。"""
    n, _t = x.shape
    pad = np.zeros((n, d - 1), dtype=x.dtype)
    xp = np.concatenate([pad, x], axis=1)
    return np.lib.stride_tricks.sliding_window_view(xp, d, axis=1).sum(axis=-1)


# ── Cross_Sectional 算子（沿 N 维，每时间步跨品种；R2.1, R2.2）────────────
#
# 输入 [N, T]：N=品种数、T=时间步。沿 dim=0 逐时间步计算，完全向量化。
# N=1 时「恒等退化」保留输入信息，避免常数污染公式搜索。出口 NaN-safe。


def _cs_rank(x: np.ndarray) -> np.ndarray:
    """每时间步跨品种百分位排名，值域 [0, 1]（R2.1）。N=1 恒等退化。"""
    n, t = x.shape
    if n == 1:
        return _n2n(x, nan=0.5, posinf=0.5, neginf=0.5)
    # 等价 torch.argsort(dim=0)：实测 torch（CPU）对相等元素保持原序，
    # 即 numpy 的 stable 排序（quicksort 在含大量并列值时会与之不一致）。
    order = np.argsort(x, axis=0, kind="stable")  # 沿 N 维排序索引
    ranks = np.empty_like(x)
    rank_vals = np.broadcast_to(np.arange(n, dtype=x.dtype).reshape(n, 1), (n, t))
    np.put_along_axis(ranks, order, rank_vals, axis=0)  # ranks[order[i,t], t] = i
    pct = ranks / (n - 1)
    return _n2n(pct, nan=0.5, posinf=0.5, neginf=0.5)


def _cs_scale(x: np.ndarray) -> np.ndarray:
    """每时间步跨品种缩放到 [0, 1]。零跨度→0.5；N=1 恒等退化。"""
    n, _t = x.shape
    if n == 1:
        return _n2n(x, nan=0.5, posinf=0.5, neginf=0.5)
    mn = x.min(axis=0, keepdims=True)  # [1, T]
    mx = x.max(axis=0, keepdims=True)  # [1, T]
    span = mx - mn
    zero_span = np.abs(span) < 1e-9
    span_safe = np.where(zero_span, np.ones_like(span), span)
    out = (x - mn) / span_safe
    out = np.where(np.broadcast_to(zero_span, out.shape), np.full_like(out, 0.5), out)
    return _n2n(out, nan=0.5, posinf=0.5, neginf=0.5)


def _cs_neutralize(x: np.ndarray) -> np.ndarray:
    """每时间步减去跨品种算术均值（截面中性化，R2.2）。N=1 恒等退化。"""
    n, _t = x.shape
    if n == 1:
        return _n2n(x, nan=0.0, posinf=0.0, neginf=0.0)
    mean = x.mean(axis=0, keepdims=True)  # [1, T]
    out = x - mean
    return _n2n(out, nan=0.0, posinf=0.0, neginf=0.0)


# ── Task 3.4 归一化 / 条件算子 helper（R2.6, R2.7）────────────────────────


def _ts_zscore(x: np.ndarray, w: int) -> np.ndarray:
    """因果滚动 z-score（R2.6）：(x - ts_mean) / (ts_std + eps)；std<eps 输出 0。"""
    windows = _ts_rolling(x, w)
    m = windows.mean(axis=-1)
    s = np.sqrt(((windows - m[..., None]) ** 2).mean(axis=-1)) + 1e-9
    z = (x - m) / s
    mask = s < (1e-9 + 1e-9)
    z = np.where(mask, np.zeros_like(z), z)
    return _n2n(z, nan=0.0, posinf=0.0, neginf=0.0)


def _winsorize(x: np.ndarray, lo: float = 0.05, hi: float = 0.95) -> np.ndarray:
    """因果滚动分位裁剪（R2.6, WINSORIZE）。

    用因果滑窗取 per-step 的 lo/hi 分位点（只用 ≤t 数据），再 clamp 当前值到
    [lower, upper]。零跨度时取原值。
    """
    w = 20
    windows = _ts_rolling(x, w)  # [N, T, w]
    # AM 用 torch.quantile(windows.float(), q, dim=-1)：float32 上做线性插值
    lower = np.quantile(windows.astype(np.float32), lo, axis=-1).astype(x.dtype)
    upper = np.quantile(windows.astype(np.float32), hi, axis=-1).astype(x.dtype)
    span = upper - lower
    safe_lower = np.where(span < 1e-9, x, lower)
    safe_upper = np.where(span < 1e-9, x, upper)
    out = np.clip(x, safe_lower, safe_upper)
    return _n2n(out, nan=0.0, posinf=0.0, neginf=0.0)


def _clip_fixed(x: np.ndarray) -> np.ndarray:
    """硬限幅 clamp(-3, 3)（R2.6, CLIP）。AM 未做 nan_to_num，此处保持一致。"""
    out: np.ndarray = np.clip(x, -3.0, 3.0)
    return out


def _sigmoid_stable(x: np.ndarray) -> np.ndarray:
    """数值稳定的 sigmoid，等价 torch.sigmoid（分段避免 float32 上溢）。"""
    out = np.empty_like(x)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[neg])
    out[neg] = ex / (1.0 + ex)
    return out


def _sigmoid_squash(x: np.ndarray) -> np.ndarray:
    """2*sigmoid(x)-1，squash 到 [-1, 1]（R2.6）。"""
    out = 2.0 * _sigmoid_stable(x) - 1.0
    return _n2n(out, nan=0.0, posinf=1.0, neginf=-1.0)


def _tanh_squash(x: np.ndarray) -> np.ndarray:
    """tanh(x)，squash 到 (-1, 1)（R2.6）。"""
    out = np.tanh(x)
    return _n2n(out, nan=0.0, posinf=1.0, neginf=-1.0)


def _if_gt(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """三元选择 where(x>0, y, z)（R2.7, IF_GT, arity 3）。输出连续值。"""
    out = np.where(x > 0, y, z)
    return _n2n(out, nan=0.0, posinf=0.0, neginf=0.0)


# ── 形状一致性校验包装（R2.9, R2.13）─────────────────────────────────────


def _with_shape_check(name: str, transform: Callable[..., Any]) -> Callable[..., Any]:
    """为二元/三元算子包装入口形状一致性校验。

    调用时先校验各操作数形状完全一致，不一致抛 :class:`ShapeError` 且不调用底层
    transform。包装后为可变位置参数形式，注册层跳过 arity 观测校验（以显式声明为准）。
    """

    def _checked(*operands: np.ndarray) -> np.ndarray:
        base = operands[0].shape
        for other in operands[1:]:
            if other.shape != base:
                raise ShapeError(
                    f"算子 '{name}' 操作数形状不一致: {tuple(base)} vs {tuple(other.shape)}"
                )
        result: np.ndarray = transform(*operands)
        return result

    return _checked


# ── 初始算子定义（保持既有 44 个算子的命名、顺序与行为）──────────────────
#
# 每项为 (name, transform, arity)。顺序即 token/算子维顺序，不可变更。

_INITIAL_OPERATORS: list[tuple[str, Callable[..., Any], int]] = [
    # ── 基础算子（token id = feat_offset+0~11）────────────────────────
    ("ADD", lambda x, y: x + y, 2),
    ("SUB", lambda x, y: x - y, 2),
    ("MUL", lambda x, y: x * y, 2),
    ("DIV", lambda x, y: x / (y + _EPS), 2),
    ("NEG", lambda x: -x, 1),
    ("ABS", np.abs, 1),
    ("SIGN", _op_sign, 1),  # torch.sign(NaN)=0，见 _op_sign
    ("GATE", _op_gate, 3),
    ("JUMP", _op_jump, 1),  # 已降低稀疏度
    ("DECAY", _op_decay, 1),
    ("DELAY1", lambda x: _ts_delay(x, 1), 1),
    ("MAX3", lambda x: np.maximum(x, np.maximum(_ts_delay(x, 1), _ts_delay(x, 2))), 1),
    # ── 时序算子（token id = feat_offset+12~21）───────────────────────
    ("TS_MEAN_5", lambda x: _ts_mean(x, 5), 1),
    ("TS_MEAN_10", lambda x: _ts_mean(x, 10), 1),
    ("TS_MEAN_20", lambda x: _ts_mean(x, 20), 1),
    ("TS_STD_5", lambda x: _ts_std(x, 5), 1),
    ("TS_STD_10", lambda x: _ts_std(x, 10), 1),
    ("TS_STD_20", lambda x: _ts_std(x, 20), 1),
    ("TS_RANK_5", lambda x: _ts_rank(x, 5), 1),
    ("TS_RANK_10", lambda x: _ts_rank(x, 10), 1),
    ("TS_RANK_20", lambda x: _ts_rank(x, 20), 1),
    ("TS_CORR_10", _ts_corr_10, 2),
    # ── 趋势 / 动量类算子（token id = feat_offset+22~27）──────────────
    ("MOMENTUM_5", lambda x: _ts_mean(x, 5) - _ts_mean(x, 20), 1),
    ("MOMENTUM_10", lambda x: _ts_mean(x, 10) - _ts_mean(x, 20), 1),
    ("TS_MAX_10", lambda x: _ts_rolling(x, 10).max(axis=-1), 1),
    ("TS_MIN_10", lambda x: _ts_rolling(x, 10).min(axis=-1), 1),
    ("WMA", _op_wma, 1),
    ("DELAY4", lambda x: _ts_delay(x, 4), 1),
    # ── v3.0 新增算子（token id = feat_offset+28~33）──────────────────
    ("EMA_5", lambda x: _ema_simple(x, 5), 1),
    ("EMA_20", lambda x: _ema_simple(x, 20), 1),
    ("TS_QUANTILE_10", lambda x: _ts_quantile(x, 10), 1),
    ("TS_SKEW_10", lambda x: _ts_skew(x, 10), 1),
    ("TS_MIN_20", lambda x: _ts_rolling(x, 20).min(axis=-1), 1),
    ("TS_MAX_20", lambda x: _ts_rolling(x, 20).max(axis=-1), 1),
    # ── v3.0 Alpha 101 + 补充算子（token id = feat_offset+34~43）──────
    ("DELTA", lambda x: _delta(x, 1), 1),
    ("TS_ARG_MAX_5", lambda x: _ts_arg_max(x, 5), 1),
    ("TS_ARG_MIN_5", lambda x: _ts_arg_min(x, 5), 1),
    ("DECAY_LINEAR_5", lambda x: _decay_linear(x, 5), 1),
    ("SCALE", lambda x: _scale(x), 1),
    ("COVARIANCE_10", lambda x, y: _ts_covariance(x, y, 10), 2),
    ("PRODUCT_5", lambda x: _ts_product(x, 5), 1),
    ("SIGNED_POWER_2", lambda x: _signed_power(x, 2.0), 1),
    ("TS_DECAY_EXP_5", lambda x: _decay_exp(x, 5, 0.5), 1),
    ("DELTA_5", lambda x: _delta(x, 5), 1),
]


# ── Task 3.2 追加：Cross_Sectional 算子（沿 N 维，每时间步跨品种）──────────
_CROSS_SECTIONAL_OPERATORS: list[tuple[str, Callable[..., Any], int]] = [
    ("CS_RANK", _cs_rank, 1),  # 跨品种百分位排名 [0,1]，N=1→恒等
    ("CS_SCALE", _cs_scale, 1),  # 跨品种缩放 [0,1]，零跨度/N=1→恒等
    ("CS_NEUTRALIZE", _cs_neutralize, 1),  # 减跨品种均值，N=1→恒等
]


# ── Task 3.3 追加：时序求和/极值与幅度变换（8 个）─────────────────────────
_TASK33_OPERATORS: list[tuple[str, Callable[..., Any], int]] = [
    # 时序求和（arity 1，因果，R2.3）
    ("TS_SUM_5", lambda x: _n2n(_ts_sum(x, 5), nan=0.0), 1),
    ("TS_SUM_10", lambda x: _n2n(_ts_sum(x, 10), nan=0.0), 1),
    ("TS_SUM_20", lambda x: _n2n(_ts_sum(x, 20), nan=0.0), 1),
    # 元素级二元极值（arity 2，天然因果，R2.4）
    ("MIN", lambda x, y: _n2n(np.minimum(x, y), nan=0.0), 2),
    ("MAX", lambda x, y: _n2n(np.maximum(x, y), nan=0.0), 2),
    # 幅度变换（arity 1，天然因果，R2.5）
    ("POWER", lambda x: _power_signed(x, 2.0), 1),
    ("SIGNED_LOG", _signed_log, 1),
    ("SQRT", _signed_sqrt, 1),
]


# ── Task 3.4 追加：归一化与条件算子（7 个；LT/GT/AND/OR 已从词表移除）──────
_TASK34_OPERATORS: list[tuple[str, Callable[..., Any], int]] = [
    # 归一化算子（arity 1，因果，R2.6）
    ("TS_ZSCORE_10", lambda x: _ts_zscore(x, 10), 1),
    ("TS_ZSCORE_20", lambda x: _ts_zscore(x, 20), 1),
    ("WINSORIZE", _winsorize, 1),
    ("CLIP", _clip_fixed, 1),
    ("SIGMOID", _sigmoid_squash, 1),
    ("TANH_SQUASH", _tanh_squash, 1),
    # 条件/逻辑算子（R2.7）：IF_GT 输出连续值，保留
    ("IF_GT", _if_gt, 3),
]


# ── 构建 OPERATOR_REGISTRY 并派生 OPS_CONFIG 导出视图 ─────────────────────

OPERATOR_REGISTRY = Registry()


def _register(registry: Registry, defs: list[tuple[str, Callable[..., Any], int]]) -> None:
    """把算子定义注册进注册表；二元/三元算子经形状校验包装（R2.13）。"""
    for name, transform, arity in defs:
        fn = _with_shape_check(name, transform) if arity >= 2 else transform
        registry.register_operator(OperatorSpec(name=name, arity=arity, transform=fn))


_register(OPERATOR_REGISTRY, _INITIAL_OPERATORS)
_register(OPERATOR_REGISTRY, _CROSS_SECTIONAL_OPERATORS)
_register(OPERATOR_REGISTRY, _TASK33_OPERATORS)
_register(OPERATOR_REGISTRY, _TASK34_OPERATORS)

# 导出视图：[(name, transform, arity), ...]，保持 AM 既有元组结构（下游兼容）
OPS_CONFIG: list[tuple[str, Callable[..., Any], int]] = [
    (spec.name, spec.transform, spec.arity) for spec in OPERATOR_REGISTRY.operator_specs
]

# 由注册表导出有序算子名视图（保持下游 import 兼容）
OPERATOR_NAMES: tuple[str, ...] = OPERATOR_REGISTRY.operator_names

# 算子总数（= 62，与冻结 AM 一致）
OPERATOR_COUNT: int = len(OPERATOR_REGISTRY.operator_names)

# 不回归断言：导出视图与注册表一致；算子总数必须为 62（顺序见 frozen_token_order.json）
assert len(OPS_CONFIG) == len(OPERATOR_REGISTRY.operator_specs), (
    "OPS_CONFIG 导出视图与 OPERATOR_REGISTRY 长度不一致"
)
assert OPERATOR_COUNT == 62, f"算子总数应为 62，实际 {OPERATOR_COUNT}"


def is_registered(name: str) -> bool:
    """该 token 名称是否已注册（跨 Feature/Operator 全局唯一）。"""
    return name in OPERATOR_REGISTRY
