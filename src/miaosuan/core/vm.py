# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""栈式虚拟机（numpy 化移植，M5）。

移植自冻结 AlphaMaster 的 ``model_core/vm.py``（原 torch 实现），逐行对齐语义：

* :class:`StackVM` —— 前缀（RPN）公式求值：``execute(formula_tokens, feat_tensor) -> [N, T]``；
* :func:`validate_formula_structure` —— 「感染模型」公式结构校验；
* :data:`POSITIVE_ONLY_OPS` / :data:`INFECTED_PROPAGATING_OPS` / :data:`SIGN_RESTORE_OPS`。

移植铁律（架构 §1.4、M5 任务书）：

* **``_normalize_output`` 原样移植**：滚动 500 期 z-score（``ddof=1``，与 ``torch.std`` 默认
  一致），``clip[-3, 3]``，warm-up 前 ``_ROLL_WINDOW-1``（=499）根输出 0；``T < 窗口`` 退化
  expanding；``N > 1`` 走截面 z-score；全局常数短路。``roll_window`` 改为**可注入参数**
  （默认 500，与 AM 一致）。
* **``except`` 不再静默**：AM 的 ``except Exception: return None`` 会吞掉真实错误。妙算仍
  **返回 None**（保持「不可求值」判定不变，否则搜索空间改变），但把异常计数并记录
  ``last_error``（实例级状态，不引入全局可变状态），保证可复现。
* **四类判定与 AM 逐点一致**：合法公式 → 数组；参数不足 → ``None``；栈深非法（残留
  ≠1）→ ``None``；NaN/Inf → ``nan_to_num(nan=0, posinf=1, neginf=-1)``。
* ``feat_offset`` / ``op_map`` / ``arity_map`` **完全动态**从词表与 ``OPS_CONFIG`` 派生。

torch → numpy 关键等价：

  ``torch.unfold(1, W, 1)``              → ``np.lib.stride_tricks.sliding_window_view``
  ``torch.std``（默认 ``unbiased=True``）→ ``np.std(..., ddof=1)``
  ``torch.clamp(v, lo, hi)``             → ``np.clip(v, lo, hi)``
  ``F.pad(x, (W-1, 0))``                 → ``np.concatenate([zeros(n, W-1), x], axis=1)``
"""

from __future__ import annotations

import warnings

import numpy as np

from .ops import OPS_CONFIG
from .vocab import FORMULA_VOCAB

__all__ = [
    "INFECTED_PROPAGATING_OPS",
    "POSITIVE_ONLY_OPS",
    "SIGN_RESTORE_OPS",
    "StackVM",
    "is_infected_propagating",
    "is_positive_only_op",
    "is_sign_restoring",
    "validate_formula_structure",
]

# 恒正算子集：输出值域非负（或几乎恒正），连续使用会丢失符号信息，
# 导致因子退化成「永远做多」的 beta 因子。
POSITIVE_ONLY_OPS: set[str] = {"TS_RANK_5", "TS_RANK_10", "TS_RANK_20", "ABS"}

# 感染传播算子：在恒正输入上输出仍恒正。
INFECTED_PROPAGATING_OPS: set[str] = {
    "TS_RANK_5", "TS_RANK_10", "TS_RANK_20", "ABS",
    "TS_SUM_5", "TS_SUM_10", "TS_SUM_20",
    "TS_MEAN_5", "TS_MEAN_10", "TS_MEAN_20",
    "TS_MAX_10", "TS_MAX_20",
    "CLIP", "SQRT", "POWER", "SIGNED_LOG",
    "SIGMOID", "TANH_SQUASH", "WINSORIZE",
    "WMA", "EMA_5", "EMA_20", "DECAY", "DECAY_LINEAR_5",
    "TS_DECAY_EXP_5",
}

# 恢复算子：能把恒正值域重新变成有正有负。
SIGN_RESTORE_OPS: set[str] = {
    "SUB", "DIV", "NEG", "GATE", "IF_GT",
    "TS_ZSCORE_10", "TS_ZSCORE_20",
    "CS_NEUTRALIZE", "CS_RANK", "CS_SCALE",
    "TS_STD_5", "TS_STD_10", "TS_STD_20",
    "TS_CORR_10", "TS_SKEW_10", "TS_QUANTILE_10",
    "DELTA", "DELTA_5", "MOMENTUM_5", "MOMENTUM_10",
}

# 默认滚动归一化窗口（与 AM ``_ROLL_WINDOW = 500`` 一致）。
DEFAULT_ROLL_WINDOW: int = 500


def is_positive_only_op(token_name: str) -> bool:
    """判断算子是否输出恒正值（可能丢失符号信息）。"""
    return token_name in POSITIVE_ONLY_OPS


def is_infected_propagating(token_name: str) -> bool:
    """判断算子是否会传播恒正感染（在恒正输入上输出仍恒正）。"""
    return token_name in INFECTED_PROPAGATING_OPS


def is_sign_restoring(token_name: str) -> bool:
    """判断算子是否能恢复符号信息（把恒正值域变回有正有负）。"""
    return token_name in SIGN_RESTORE_OPS


def validate_formula_structure(
    formula_tokens: list[int], vocab_names: tuple[str, ...]
) -> list[str]:
    """校验公式结构，返回违规原因列表（空列表 = 合法）。

    使用「感染模型」：一旦公式中出现恒正算子（如 TS_RANK），后续如果连续使用传播算子
    （如 TS_SUM/TS_MEAN/CLIP/SQRT），值域会一直保持非负，导致因子退化为 beta。只有恢复
    算子（如 SUB/TS_ZSCORE/CS_NEUTRALIZE）才能打破感染。

    规则：
    1. 禁止恒正算子后连续 2 个以上传播算子（感染链太长）；
    2. 公式末尾若仍处于感染状态（恒正且未恢复），标记违规。
    """
    violations: list[str] = []
    feat_offset = FORMULA_VOCAB.operator_offset

    infected = False  # 当前值域是否已被感染（恒正）
    infected_chain_len = 0  # 感染链长度
    last_positive_op: str | None = None  # 最后一个引发感染的算子名

    for i, token in enumerate(formula_tokens):
        token = int(token)
        if token < feat_offset:
            # 特征 token：不改变感染状态
            continue
        name = vocab_names[token] if token < len(vocab_names) else f"op_{token}"

        if is_positive_only_op(name):
            # 恒正算子：开始/延续感染
            if not infected:
                infected = True
                last_positive_op = name
            infected_chain_len += 1
        elif infected and is_sign_restoring(name):
            # 恢复算子：打破感染
            infected = False
            infected_chain_len = 0
            last_positive_op = None
        elif infected and is_infected_propagating(name):
            # 传播算子：感染继续
            infected_chain_len += 1
            if infected_chain_len >= 3:
                violations.append(
                    f"步骤{i}: 恒正感染链过长（从 {last_positive_op} 起 "
                    f"{infected_chain_len} 个传播算子），因子将退化为 beta"
                )
        # else: 非感染相关算子（如 ADD/MUL），不改变感染状态

    if infected and infected_chain_len >= 2:
        violations.append(
            f"公式末尾处于恒正感染状态（链长 {infected_chain_len}），因子输出将偏向单方向"
        )

    return violations


class StackVM:
    """前缀公式栈式虚拟机（AM ``StackVM`` 的 numpy 移植，行为等价）。"""

    def __init__(self, roll_window: int = DEFAULT_ROLL_WINDOW) -> None:
        # feat_offset 动态从 FORMULA_VOCAB.operator_offset 读取（= feature_count = F）。
        self.feat_offset = FORMULA_VOCAB.operator_offset
        # op_map / arity_map 动态从 OPS_CONFIG 构建。
        self.op_map = {i + self.feat_offset: cfg[1] for i, cfg in enumerate(OPS_CONFIG)}
        self.arity_map = {i + self.feat_offset: cfg[2] for i, cfg in enumerate(OPS_CONFIG)}
        # 恒正算子 token id 集合（用于采样时约束）。
        self.positive_only_ids: set[int] = {
            i + self.feat_offset
            for i, cfg in enumerate(OPS_CONFIG)
            if cfg[0] in POSITIVE_ONLY_OPS
        }
        # 归一化滚动窗口（可注入；默认 500 与 AM 一致）。
        self.roll_window = int(roll_window)
        # 实例级统计（不引入全局可变状态，保证可复现）：不可求值计数 + 末次异常。
        self.none_count = 0
        self.error_count = 0
        self.last_error: str | None = None

    @staticmethod
    def _normalize_output(
        x: np.ndarray, roll_window: int = DEFAULT_ROLL_WINDOW
    ) -> np.ndarray:
        """对因子输出做因果标准化（AM ``_normalize_output`` 原样移植）。

        策略（两级降级，全部因果）：

        1. 全局常数短路（``std < 1e-6``）→ 原样返回（由 engine 的 const 拦截）；
        2. 截面 z-score（``N > 1``，每时间步跨品种，``ddof=1``）；
        3. 滚动时序 z-score（``N == 1``，窗口 ``roll_window``，``ddof=1``）：
           ``T < 窗口`` 退化 expanding；warm-up 前 ``窗口-1`` 根输出 0。

        Returns:
            ``[N, T]`` clip 到 ``[-3, 3]``；常数则原样返回。
        """
        n, t = x.shape

        # 检测是否是全局常数（标准化无意义）。
        global_std = float(x.std(ddof=1)) if t > 1 else 0.0
        if global_std < 1e-6:
            return x  # 常数因子，由 engine 的 const_cnt 拦截

        # ── 截面标准化（跨品种，每时间步；N=1 时跳过）──────────────
        if n > 1:
            cs_mean = x.mean(axis=0, keepdims=True)
            cs_std = np.clip(x.std(axis=0, keepdims=True, ddof=1), 1e-8, None)
            cs_z = (x - cs_mean) / cs_std
            return np.clip(cs_z, -3.0, 3.0)

        # ── 滚动时序标准化（每品种独立，固定窗口，无 look-ahead）─────
        _roll = int(roll_window)

        if t < _roll:
            # 样本不足：退化为 expanding z-score（仍因果，无 look-ahead）
            cnt = np.arange(1, t + 1, dtype=x.dtype).reshape(1, t)
            cumsum = np.cumsum(x, axis=1)
            ts_mean = cumsum / cnt
            cumsum_sq = np.cumsum(x * x, axis=1)
            ts_var = (cumsum_sq / cnt) - ts_mean * ts_mean
            ts_std = np.sqrt(np.clip(ts_var, 1e-8, None))
            ts_z = (x - ts_mean) / ts_std
            return np.asarray(np.clip(ts_z, -3.0, 3.0))

        # 滚动均值/标准差（窗口 _roll，因果：只用 [t-W+1, t]）。
        pad = np.zeros((n, _roll - 1), dtype=x.dtype)
        padded = np.concatenate([pad, x], axis=1)  # [N, T+W-1]
        windows = np.lib.stride_tricks.sliding_window_view(padded, _roll, axis=1)  # [N, T, W]
        ts_mean = windows.mean(axis=2)  # [N, T]
        ts_std = np.clip(windows.std(axis=2, ddof=1), 1e-8, None)  # [N, T]
        ts_z = (x - ts_mean) / ts_std  # [N, T]

        # warm-up 期（前 _roll-1 根）输出 0（因子中性，不出信号）。
        ts_z[:, np.arange(t) < (_roll - 1)] = 0.0

        return np.asarray(np.clip(ts_z, -3.0, 3.0))

    def execute(self, formula_tokens: list[int], feat_tensor: np.ndarray) -> np.ndarray | None:
        """执行前缀公式，返回 ``[N, T]`` 因子或 ``None``（不可求值）。

        判定与 AM 逐点一致（四类）：

        * 合法公式（栈恰好剩 1 个元素）→ 归一化后的因子数组；
        * 参数不足（栈元素 < 算子 arity）→ ``None``；
        * 栈深非法（结束时残留 ≠ 1）→ ``None``；
        * 特征 token 越界、算子在 op_map 外、或异常 → ``None``。

        算子输出若含 NaN/Inf，按 AM 语义 ``nan_to_num(nan=0.0, posinf=1.0, neginf=-1.0)``。
        """
        stack: list[np.ndarray] = []
        try:
            for token in formula_tokens:
                token = int(token)
                if token < self.feat_offset:
                    if token >= feat_tensor.shape[1]:
                        self.none_count += 1
                        return None
                    stack.append(feat_tensor[:, token, :])
                elif token in self.op_map:
                    arity = self.arity_map[token]
                    if len(stack) < arity:
                        self.none_count += 1
                        return None
                    args: list[np.ndarray] = []
                    for _ in range(arity):
                        args.append(stack.pop())
                    args.reverse()
                    res = self.op_map[token](*args)
                    if np.isnan(res).any() or np.isinf(res).any():
                        res = np.nan_to_num(res, nan=0.0, posinf=1.0, neginf=-1.0)
                    stack.append(res)
                else:
                    self.none_count += 1
                    return None
            if len(stack) == 1:
                return self._normalize_output(stack[0], self.roll_window)
            self.none_count += 1
            return None
        except Exception as exc:  # noqa: BLE001 - 保持「返回 None」语义，但不再静默吞错
            self.error_count += 1
            self.last_error = f"{type(exc).__name__}: {exc}"
            warnings.warn(
                f"StackVM.execute 捕获异常，已返回 None（保持 AM 语义）：{self.last_error}",
                RuntimeWarning,
                stacklevel=2,
            )
            return None
