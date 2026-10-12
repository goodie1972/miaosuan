# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""G2 对拍 harness（路线图 R2.4，闭环正确性的验收基石）。

同一份已收盘 bars：**回测仓位序列 == 实时信号仓位序列（逐位相等）**。

两侧唯一允许的公共来源：``core.vm`` + ``Semantics``——任何一侧私自引入
『未收盘 bar / 不同 lag / 不同中性带 / 不同滚动窗』都会在此变红。
研究侧口径锚定 :func:`miaosuan.report.equity.run_full_backtest` 的计算链
（features → StackVM → tanh + 中性带），运行时侧为
:class:`~miaosuan.runtime.signal_engine.SignalEngine` 逐 bar 重算取末列。

注：``tests/parity/`` 是**研究侧**对拍（妙算 vs 冻结基准），本文件是
**运行时侧**对拍（runtime vs 回测），勿混淆。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from miaosuan.core.features import compute_features
from miaosuan.core.signal import compute_target_positions
from miaosuan.core.vm import StackVM
from miaosuan.core.vocab import VOCAB_VERSION
from miaosuan.ir.schema import (
    Evidence,
    FactorPayload,
    Semantics,
    StrategySpec,
)
from miaosuan.runtime import SignalEngine
from miaosuan.runtime.ports import Bar
from miaosuan.search.rpn import random_feasible

_TIMEFRAME = "M30"
_TF_SECONDS = 1800
_N_BARS = 700  # > roll_window(500)，覆盖 warm-up/expanding/rolling 三段


def make_spec(seed: int = 42) -> StrategySpec:
    """确定性生成一个**可求值**的因子 spec（random_feasible 保证栈深合法）。"""
    rng = np.random.default_rng(seed)
    tokens = tuple(int(t) for t in random_feasible(rng, 8))
    return StrategySpec(
        name="parity_smoke",
        payload=FactorPayload(tokens=tokens, vocab_version=VOCAB_VERSION),
        semantics=Semantics(timeframe=_TIMEFRAME),
        evidence=Evidence(gate_verdict="DEPLOYABLE"),
    )


def make_bars(seed: int = 7, n: int = _N_BARS, *, end_ts: int = 1_800_000_000) -> list[Bar]:
    """随机游走 OHLCV bars（确定性种子；ts 升序、周期连续）。"""
    rng = np.random.default_rng(seed)
    close = 4000.0 + np.cumsum(rng.normal(0.0, 3.0, n))
    open_ = np.empty_like(close)
    open_[0] = close[0]
    open_[1:] = close[:-1] + rng.normal(0.0, 0.5, n - 1)
    high = np.maximum(open_, close) + rng.uniform(0.5, 3.0, n)
    low = np.minimum(open_, close) - rng.uniform(0.5, 3.0, n)
    volume = rng.integers(50, 500, n).astype(np.float64)
    ts = end_ts - (n - 1) * _TF_SECONDS + np.arange(n) * _TF_SECONDS
    return [
        Bar(
            symbol="XAUUSD",
            timeframe=_TIMEFRAME,
            ts=int(ts[i]),
            open=float(open_[i]),
            high=float(high[i]),
            low=float(low[i]),
            close=float(close[i]),
            volume=float(volume[i]),
            closed=True,
        )
        for i in range(n)
    ]


def backtest_positions(spec: StrategySpec, bars: list[Bar]) -> np.ndarray:
    """研究侧参考实现：与 ``run_full_backtest`` 同链（features → VM → tanh+带）。"""
    raw: dict[str, Any] = {
        "open": np.asarray([[b.open for b in bars]], dtype=np.float64),
        "high": np.asarray([[b.high for b in bars]], dtype=np.float64),
        "low": np.asarray([[b.low for b in bars]], dtype=np.float64),
        "close": np.asarray([[b.close for b in bars]], dtype=np.float64),
        "volume": np.asarray([[b.volume for b in bars]], dtype=np.float64),
    }
    features = compute_features(raw)  # [1, F, T]
    vm = StackVM(roll_window=int(spec.semantics.roll_window))
    factor = vm.execute([int(t) for t in spec.payload.tokens], features)  # type: ignore[union-attr]
    assert factor is not None, "对拍 spec 必须可求值（random_feasible 保证）"
    neutral_band = float(spec.semantics.neutral_band or 0.0) or 0.05
    positions = compute_target_positions(
        factor,
        min_trade_exposure=neutral_band,
        long_only=not spec.semantics.long_short,
    )
    return positions[0]


def test_live_signal_matches_backtest_bar_by_bar(spec_fixture: StrategySpec) -> None:
    """核心断言：回测仓位序列 == 实时信号仓位序列（逐位相等）。"""
    bars = make_bars()
    bt_positions = backtest_positions(spec_fixture, bars)

    engine = SignalEngine(spec_fixture)
    rt_positions = list(engine.replay_closed_bars(bars))

    assert rt_positions == pytest.approx([float(p) for p in bt_positions], abs=0.0)
    # 对拍有效性：warm-up 后必须有非零仓位（否则序列全 0，测试失去意义）
    warmup = int(spec_fixture.semantics.roll_window) - 1
    assert any(p != 0.0 for p in rt_positions[warmup:])


def test_replay_is_deterministic_across_restarts(spec_fixture: StrategySpec) -> None:
    """同一份 bars 重放两次（模拟重启后按历史回填）结果逐位一致。"""
    bars = make_bars()
    first = list(SignalEngine(spec_fixture).replay_closed_bars(bars))
    second = list(SignalEngine(spec_fixture).replay_closed_bars(bars))
    assert first == second


def test_partial_replay_tail_matches_full_replay(spec_fixture: StrategySpec) -> None:
    """先喂前 K 根再继续喂剩余（模拟分段回填/续拉）与一次性全喂位等。"""
    bars = make_bars()
    full = list(SignalEngine(spec_fixture).replay_closed_bars(bars))

    k = len(bars) // 2
    engine = SignalEngine(spec_fixture)
    part1 = list(engine.replay_closed_bars(bars[:k]))
    part2 = list(engine.replay_closed_bars(bars[k:]))

    assert part1 + part2 == full


@pytest.fixture
def spec_fixture() -> StrategySpec:
    return make_spec()
