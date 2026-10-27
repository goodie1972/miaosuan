# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for full license text.

"""runtime 信号引擎：StrategySpec → 已收盘 bars → StackVM → 仓位（路线图 R2.1）。

与回测引擎**共用同一口径**（G2 对拍的基石）::

    compute_features(raw_dict)  →  StackVM(roll_window, gate_deadband).execute(tokens, feats)
    →  tanh(factor) → |pos| < neutral_band 置 0（long_only 时负仓位归零）

全部口径来自 ``Semantics``（``neutral_band`` / ``roll_window`` / ``gate_deadband`` /
``long_short`` / ``execution_lag_bars``），运行时不允许另立配置。逐 bar 语义：bar t
收盘时引擎产出 ``position[t]``——按 ``execution_lag_bars=1`` 于 t+1 开盘可执行，与
回测的 ``target_ret[t] = log(open[t+2]/open[t+1])`` 收益对齐。

启动安全链（Sample G，顺序不可调换）：门禁 verdict → 词表版本 → 才允许挂载。
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np

from ..core.features import compute_features
from ..core.signal import MIN_TRADE_EXPOSURE, compute_target_positions
from ..core.vm import StackVM
from ..core.vocab import FORMULA_VOCAB
from ..errors import RuntimeNotDeployableError, RuntimeSignalError
from ..ir.codec import read_spec
from ..ir.schema import FactorPayload, StrategySpec
from .ports import Bar, NotifierPort

__all__ = ["SignalEngine", "load_spec_checked", "engine_signal_summary"]

#: 允许的仓位映射函数（``Semantics.position_fn``）；出现新映射须同步回测侧。
_ALLOWED_POSITION_FNS = frozenset({"tanh"})


def load_spec_checked(path: Path) -> StrategySpec:
    """启动安全链（Sample G）：读 spec → 门禁硬校验 → 词表版本校验。

    顺序不可调换；任何一步不过都不允许进入 runtime（实盘通道只吃已证明
    可部署、且词表匹配的策略）。门禁结论存于 ``Evidence.gate_verdict``
    （路线图 Sample G 曾误写为 provenance，落地时以 schema 为准）。
    """
    spec = read_spec(path)
    if spec.evidence.gate_verdict != "DEPLOYABLE":
        raise RuntimeNotDeployableError(
            f"策略未过部署门禁: gate_verdict={spec.evidence.gate_verdict!r}"
            f"（仅 DEPLOYABLE 可进 runtime）",
            context={"spec_id": spec.spec_id, "verdict": spec.evidence.gate_verdict},
        )
    if not isinstance(spec.payload, FactorPayload):
        raise RuntimeSignalError(
            f"runtime 仅支持因子载荷（得到 {type(spec.payload).__name__}）",
            context={"spec_id": spec.spec_id},
        )
    FORMULA_VOCAB.verify(spec.payload.vocab_version)  # E-VOCAB-MISMATCH
    return spec


class SignalEngine:
    """单品种信号引擎：吃已收盘 Bar，逐根产出目标仓位 [-1, +1]。

    每根新收盘 bar 触发一次「全历史重算、取末列」：与面板级回测在相同
    数据上逐列位等（滚动窗/特征全部因果，见 ``tests/parity`` 的因果性
    测试），这是 G2 对拍成立的实现基础。

    :param spec: 已通过 :func:`load_spec_checked`（或等价校验）的因子 spec。
    :param notifier: 可选通知端口（转折推送等由调用层接线，引擎本身不通知）。
    """

    def __init__(self, spec: StrategySpec, *, notifier: NotifierPort | None = None) -> None:
        payload = spec.payload
        if not isinstance(payload, FactorPayload):
            raise RuntimeSignalError(
                f"runtime 仅支持因子载荷（得到 {type(payload).__name__}）",
                context={"spec_id": spec.spec_id},
            )
        sem = spec.semantics
        if sem.position_fn not in _ALLOWED_POSITION_FNS:
            raise RuntimeSignalError(
                f"不支持的仓位映射函数: {sem.position_fn!r}",
                context={"allowed": sorted(_ALLOWED_POSITION_FNS)},
            )
        self._spec = spec
        self._tokens = [int(t) for t in payload.tokens]
        # 中性带映射与 cli 回测预览同口径：0（未设）→ 默认 0.05
        self._neutral_band = float(sem.neutral_band or 0.0) or MIN_TRADE_EXPOSURE
        self._long_only = not sem.long_short
        self._vm = StackVM(roll_window=int(sem.roll_window), gate_deadband=float(sem.gate_deadband))
        self._notifier = notifier
        # 历史缓冲（[1, T] 单品种）；ts → 已处理收盘 bar 水位（去重）
        self._history: dict[str, list[float]] = {
            "open": [],
            "high": [],
            "low": [],
            "close": [],
            "volume": [],
        }
        self._last_ts: int | None = None
        self._last_position: float = 0.0

    # ── 属性 ──────────────────────────────────────────────────────────────

    @property
    def spec(self) -> StrategySpec:
        return self._spec

    @property
    def last_position(self) -> float:
        return self._last_position

    @property
    def bars_seen(self) -> int:
        """已消费的收盘 bar 数（观测用：回填/续跑进度）。"""
        return len(self._history["close"])

    # ── 核心语义 ──────────────────────────────────────────────────────────

    def on_closed_bar(self, bar: Bar) -> float:
        """消费一根已收盘 bar，返回该 bar 收盘时的目标仓位。

        去重：``ts`` 不严格递增的重复 bar 直接返回上次的仓位（不重算）；
        ``closed`` 不为真的 bar 一律拒绝（收盘纪律）。
        """
        if not bar.closed:
            raise RuntimeSignalError(
                "信号引擎只吃已收盘 bar（收盘纪律）",
                context={"symbol": bar.symbol, "ts": bar.ts},
            )
        if self._last_ts is not None and bar.ts <= self._last_ts:
            return self._last_position  # 重复/回退 bar：幂等返回
        self._append(bar)
        self._last_ts = bar.ts
        self._last_position = self._compute_last_position()
        return self._last_position

    def replay_closed_bars(self, bars: Iterable[Bar]) -> Iterator[float]:
        """按序回放一批已收盘 bar（对拍 harness 的运行时侧入口）。"""
        for bar in bars:
            yield self.on_closed_bar(bar)

    # ── 内部 ──────────────────────────────────────────────────────────────

    def _append(self, bar: Bar) -> None:
        self._history["open"].append(float(bar.open))
        self._history["high"].append(float(bar.high))
        self._history["low"].append(float(bar.low))
        self._history["close"].append(float(bar.close))
        self._history["volume"].append(float(bar.volume))

    def _compute_last_position(self) -> float:
        """全历史重算特征/因子/仓位，取末列（与面板级回测逐列位等）。

        预热门控：历史不足 ``warmup_bars + 1`` 根时直接返回 0——与面板级
        计算的 warm-up 置零规则（前 ``roll_window-1`` 列输出 0，``StackVM.
        _normalize_output``）同口径。这是 G2 对拍成立的必要条件：面板级
        预热段恒为 0，引擎若在短历史上走 expanding 归一化会产生非零值，
        两侧在预热段必然漂移。
        """
        n = len(self._history["close"])
        if n <= int(self._spec.semantics.warmup_bars):
            return 0.0
        raw = {k: np.asarray([v], dtype=np.float64) for k, v in self._history.items()}
        features = compute_features(raw)  # [1, F, T]
        factor = self._vm.execute(self._tokens, features)
        if factor is None:
            raise RuntimeSignalError(
                "公式不可求值（StackVM 返回 None）",
                context={"spec_id": self._spec.spec_id, "tokens": self._tokens},
            )
        positions = compute_target_positions(
            factor,
            min_trade_exposure=self._neutral_band,
            long_only=self._long_only,
        )
        return float(positions[0, -1])


def engine_signal_summary(engine: SignalEngine, position: float) -> str:
    """人类可读的信号摘要（日志/通知共用；不含持仓金额，脱敏口径见 R5.2）。"""
    direction = "多" if position > 0 else ("空" if position < 0 else "观望")
    spec = engine.spec
    return f"{spec.name} pos={position:+.4f} ({direction}) lag={spec.semantics.execution_lag_bars}"
