# -*- coding: utf-8 -*-
# Copyright (c) 2026 Original Author
# Copyright (c) 2026 Your Company
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""信号映射（numpy 化移植，M5）—— 连续仓位 + long_only 开关。

移植自冻结 AlphaMaster 的 ``strategy_manager/signal.py``（原 torch 实现），逐行对齐其
数值语义（收益优先的连续仓位模式）：

    position = tanh(factor)  →  中性带（|pos| < band ⇒ 0）  →  long_only 开关

关键等价与参数化：

* AM 由根目录 ``config.Config.MIN_TRADE_EXPOSURE``（=0.05）隐式读取阈值；妙算改为
  **参数注入** ``min_trade_exposure``（默认 0.05，与 AM 一致），``core/`` 不读环境、不做 IO。
* AM 的 ``torch.tanh`` → ``np.tanh``；``torch.where(cond,a,b)`` → ``np.where(cond,a,b)``；
  ``torch.zeros_like`` → ``np.zeros_like``。dtype 保持（float32 in → float32 out）。
* ``long_only`` 为妙算新增的**预留开关**（架构 §T02 验收 4：long_only=True 时无负仓位），
  默认 False 以保持与 AM 的逐点一致；置 True 时先把负仓位归零，再套中性带。

保留 AM 的实盘阈值常量与动作常量，供下游（adapters / 实盘 runner）读取。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..errors import ConfigError

__all__ = [
    "CLOSE",
    "ENTRY_THRESHOLD",
    "EXIT_THRESHOLD",
    "HOLD",
    "MIN_TRADE_EXPOSURE",
    "OPEN_LONG",
    "OPEN_SHORT",
    "REVERSE_TO_LONG",
    "REVERSE_TO_SHORT",
    "SignalMapper",
    "compute_target_positions",
    "compute_target_positions_stateless",
    "reconcile_action",
    "target_to_direction",
]

# ── 保留实盘用的阈值参数（实盘 Runner 可能还读取这些常量）──────────────────
ENTRY_THRESHOLD: float = 0.3
EXIT_THRESHOLD: float = 0.1
MIN_TRADE_EXPOSURE: float = 0.05

# ── 动作常量 ──────────────────────────────────────────────────────────────
HOLD: str = "HOLD"
OPEN_LONG: str = "OPEN_LONG"
OPEN_SHORT: str = "OPEN_SHORT"
CLOSE: str = "CLOSE"
REVERSE_TO_LONG: str = "REVERSE_TO_LONG"
REVERSE_TO_SHORT: str = "REVERSE_TO_SHORT"


def compute_target_positions(
    factors: np.ndarray,
    prev_positions: np.ndarray | None = None,
    *,
    min_trade_exposure: float = MIN_TRADE_EXPOSURE,
    long_only: bool = False,
) -> np.ndarray:
    """把因子张量转换为连续仓位 [-1, +1]（收益优先模式）。

    等价 AM ``compute_target_positions``：``pos = tanh(factors)``，再对
    ``|pos| < min_trade_exposure`` 的位置置 0。

    Args:
        factors: ``[N, T]`` 或 ``[N]`` 的因子张量。
        prev_positions: 保留参数，连续模式下忽略（AM 兼容）。
        min_trade_exposure: 中性带下界（参数注入；默认 0.05，与 AM Config 一致）。
        long_only: True 时负仓位归零（预留开关），默认 False。

    Returns:
        与 ``factors`` 同形状的连续仓位数组。
    """
    pos = np.tanh(factors)
    if long_only:
        pos = np.where(pos > 0.0, pos, np.zeros_like(pos))
    if min_trade_exposure > 0.0:
        pos = np.where(np.abs(pos) >= min_trade_exposure, pos, np.zeros_like(pos))
    return pos


def compute_target_positions_stateless(
    factors: np.ndarray,
    *,
    min_trade_exposure: float = MIN_TRADE_EXPOSURE,
    long_only: bool = False,
) -> np.ndarray:
    """无状态版本（供训练 / 回测快速计算，连续仓位模式）。"""
    return compute_target_positions(
        factors,
        prev_positions=None,
        min_trade_exposure=min_trade_exposure,
        long_only=long_only,
    )


def target_to_direction(target: float, min_abs: float | None = None) -> int:
    """把连续目标仓位转成 MT5 可执行方向（+1 / -1 / 0）。"""
    threshold = MIN_TRADE_EXPOSURE if min_abs is None else float(min_abs)
    if target >= threshold:
        return 1
    if target <= -threshold:
        return -1
    return 0


def reconcile_action(current: int, target: int) -> str:
    """根据当前仓位方向和目标方向，返回应执行的动作字符串。

    Args:
        current: 当前仓位方向，+1（多）/ -1（空）/ 0（空仓）。
        target: 目标仓位方向，+1 / -1 / 0。

    Returns:
        动作字符串，取值为模块级常量之一。
    """
    if current == target:
        return HOLD
    if current == 0:
        return OPEN_LONG if target == 1 else OPEN_SHORT
    if target == 0:
        return CLOSE
    return REVERSE_TO_LONG if target == 1 else REVERSE_TO_SHORT


@dataclass(frozen=True)
class SignalMapper:
    """信号映射器（架构类图 `SignalMapper`）。

    把「因子 → 连续仓位」这一步封装为可配置对象，便于 search / backtest 复用同一映射，
    并预留 ``long_only`` / ``neutral_band`` 两个产品轴开关（默认与 AM 一致）。

    :param position_fn: 仓位函数名，当前仅支持 ``"tanh"``（AM 语义）。
    :param neutral_band: 中性带下界（映射到 ``min_trade_exposure``）。
    :param long_only: 是否只做多（负仓位归零）。
    """

    position_fn: str = "tanh"
    neutral_band: float = MIN_TRADE_EXPOSURE
    long_only: bool = False

    def to_position(self, factor: np.ndarray) -> np.ndarray:
        """把因子映射为连续仓位。"""
        if self.position_fn != "tanh":
            raise ConfigError(
                f"不支持的 position_fn={self.position_fn!r}（仅支持 'tanh'）",
                context={"position_fn": self.position_fn},
            )
        return compute_target_positions(
            factor,
            min_trade_exposure=self.neutral_band,
            long_only=self.long_only,
        )
