# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for full license text.

"""端到端模拟测试（R1 ↔ R2 全链路，不依赖真实 MT5 终端）。

模拟数据流::

    假 MT5 终端 → DataAcquisition.fetch(source="MT5") → parquet 缓存
    → loader.load 读回 Panel → 回测参考链（features → StackVM → 仓位）
    vs SignalEngine 逐 bar 重算 —— 两侧逐位相等（G2 在真实落盘链路上的复验）

与 ``test_runtime_signal_parity.py``（内存 bars 直喂）互补：本文件验证
parquet 序列化往返、loader 契约、文件名推断这条**真实数据通路**同样
不破坏对拍，并复验 G1 命名契约 + 增量更新后的信号一致性。
"""

from __future__ import annotations

import sys
import time
import types
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from miaosuan.core.features import compute_features
from miaosuan.core.signal import compute_target_positions
from miaosuan.core.vm import StackVM
from miaosuan.core.vocab import VOCAB_VERSION
from miaosuan.data.acquisition import DataAcquisition, FetchStatus, TypedNetworkSource
from miaosuan.data.fetchers import MT5Fetcher
from miaosuan.data.loader import load
from miaosuan.ir.schema import Evidence, FactorPayload, Semantics, StrategySpec
from miaosuan.runtime import SignalEngine
from miaosuan.runtime.ports import Bar
from miaosuan.search.rpn import random_feasible

_TF = "M30"
_TF_SECONDS = 1800
_SERVER_OFFSET_HOURS = 2.0
_N_BARS = 700

# 复用假终端装置（与 test_fetchers_mt5.py 同形态，自持一份避免跨测试文件耦合）
_RATES_DTYPE = np.dtype(
    [
        ("time", "<i8"),
        ("open", "<f8"),
        ("high", "<f8"),
        ("low", "<f8"),
        ("close", "<f8"),
        ("tick_volume", "<i8"),
        ("spread", "<i4"),
        ("real_volume", "<i8"),
    ]
)


class _FakeMT5Module(types.ModuleType):
    TIMEFRAME_M30 = 30

    def __init__(self, store: dict[tuple[str, str], list[tuple]], *, offset_hours: float) -> None:
        super().__init__("MetaTrader5")
        self._store = store
        self._offset = offset_hours

    def initialize(self, **kwargs: Any) -> bool:
        return True

    def shutdown(self) -> None:
        return None

    def last_error(self) -> tuple[int, str]:
        return (0, "ok")

    def symbol_select(self, symbol: str, enable: bool) -> bool:
        return True

    def symbol_info_tick(self, symbol: str) -> SimpleNamespace:
        return SimpleNamespace(time=int(time.time()) + int(self._offset * 3600))

    def copy_rates_from_pos(
        self, symbol: str, timeframe: int, start_pos: int, count: int
    ) -> np.ndarray:
        bars = self._store.get((symbol, "M30"), [])
        if start_pos != 0 or not bars:
            return np.zeros(0, dtype=_RATES_DTYPE)
        return np.array(bars[-count:], dtype=_RATES_DTYPE)


def make_m30_bars(n: int, *, end_utc: int, seed: int = 11) -> list[tuple]:
    """随机游走 M30 bars（raw = utc + 服务器偏移）；确定性。"""
    rng = np.random.default_rng(seed)
    close = 4000.0 + np.cumsum(rng.normal(0.0, 3.0, n))
    open_ = np.empty_like(close)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    high = np.maximum(open_, close) + rng.uniform(0.5, 3.0, n)
    low = np.minimum(open_, close) - rng.uniform(0.5, 3.0, n)
    volume = rng.integers(50, 500, n).astype(np.float64)
    end_raw = end_utc + int(_SERVER_OFFSET_HOURS * 3600)
    return [
        (
            int(end_raw - (n - 1 - i) * _TF_SECONDS),
            float(open_[i]),
            float(high[i]),
            float(low[i]),
            float(close[i]),
            int(volume[i]),
            0,
            0,
        )
        for i in range(n)
    ]


def make_spec(seed: int = 42) -> StrategySpec:
    rng = np.random.default_rng(seed)
    tokens = tuple(int(t) for t in random_feasible(rng, 8))
    return StrategySpec(
        name="e2e_smoke",
        payload=FactorPayload(tokens=tokens, vocab_version=VOCAB_VERSION),
        semantics=Semantics(timeframe=_TF),
        evidence=Evidence(gate_verdict="DEPLOYABLE"),
    )


def test_fetch_to_signal_end_to_end(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """R1→R2 全链路：落盘 → 读回 → 回测参考 vs runtime 逐位对拍。"""
    from miaosuan.data.acquisition import _cache_dirs_from_config
    from miaosuan.config import kline_data_dir
    monkeypatch.setattr("miaosuan.data.acquisition._cache_dirs_from_config", lambda: (tmp_path,))
    monkeypatch.setattr("miaosuan.config.kline_data_dir", lambda: str(tmp_path))
    end_utc = int(time.time()) - 3600  # 全部已收盘
    store = {("XAUUSD", _TF): make_m30_bars(_N_BARS, end_utc=end_utc)}
    monkeypatch.setitem(
        sys.modules, "MetaTrader5", _FakeMT5Module(store, offset_hours=_SERVER_OFFSET_HOURS)
    )

    acq = DataAcquisition(
        sources=[TypedNetworkSource(MT5Fetcher(utc_offset_hours=_SERVER_OFFSET_HOURS))]
    )
    res = acq.fetch("XAUUSD", _TF, source="MT5")
    assert res.status is FetchStatus.UPDATED
    assert res.path.name == f"XAUUSD_{_TF}_MT5.parquet"  # G1 命名契约

    # ── 真实落盘链路读回 ─────────────────────────────────────────────
    panel = load(res.path)
    assert panel.fields["close"].shape == (1, _N_BARS)
    raw = panel.to_raw_dict()
    spec = make_spec()

    features = compute_features(raw)
    factor = StackVM(roll_window=int(spec.semantics.roll_window)).execute(
        [int(t) for t in spec.payload.tokens],
        features,  # type: ignore[union-attr]
    )
    assert factor is not None
    neutral_band = float(spec.semantics.neutral_band or 0.0) or 0.05
    bt = compute_target_positions(
        factor, min_trade_exposure=neutral_band, long_only=not spec.semantics.long_short
    )[0]

    bars = [
        Bar(
            symbol="XAUUSD",
            timeframe=_TF,
            ts=int(panel.time[i]),
            open=float(panel.fields["open"][0, i]),
            high=float(panel.fields["high"][0, i]),
            low=float(panel.fields["low"][0, i]),
            close=float(panel.fields["close"][0, i]),
            volume=float(panel.fields["volume"][0, i]),
            closed=True,
        )
        for i in range(panel.fields["close"].shape[1])
    ]

    rt = list(SignalEngine(spec).replay_closed_bars(bars))
    assert rt == pytest.approx([float(p) for p in bt], abs=0.0)
    warmup = int(spec.semantics.warmup_bars)
    assert any(p != 0.0 for p in rt[warmup:])


def test_incremental_append_keeps_signal_consistency(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """增量追加 3 根后重拉：全量重算与「首次全量 + 增量」的仓位序列一致。"""
    from miaosuan.data.acquisition import _cache_dirs_from_config
    from miaosuan.config import kline_data_dir
    monkeypatch.setattr("miaosuan.data.acquisition._cache_dirs_from_config", lambda: (tmp_path,))
    monkeypatch.setattr("miaosuan.config.kline_data_dir", lambda: str(tmp_path))
    end_utc = int(time.time()) - 3600
    bars = make_m30_bars(_N_BARS, end_utc=end_utc)
    mod = _FakeMT5Module({("XAUUSD", _TF): bars}, offset_hours=_SERVER_OFFSET_HOURS)
    monkeypatch.setitem(sys.modules, "MetaTrader5", mod)

    acq = DataAcquisition(
        sources=[TypedNetworkSource(MT5Fetcher(utc_offset_hours=_SERVER_OFFSET_HOURS))]
    )
    first = acq.fetch("XAUUSD", _TF, source="MT5")
    assert first.status is FetchStatus.UPDATED

    # 追加 3 根新收盘 bar（时间延续）
    price = float(bars[-1][4])
    for i in range(1, 4):
        t = int(bars[-1][0]) + _TF_SECONDS * i
        price += 0.5
        bars.append((t, price, price + 1.0, price - 1.0, price + 0.2, 100 + i, 0, 0))

    second = acq.fetch("XAUUSD", _TF, source="MT5")
    assert second.status is FetchStatus.UPDATED
    assert second.path == first.path

    panel = load(second.path)
    assert panel.fields["close"].shape == (1, _N_BARS + 3), "增量 3 根并入缓存"

    spec = make_spec()
    features = compute_features(panel.to_raw_dict())
    factor = StackVM(roll_window=int(spec.semantics.roll_window)).execute(
        [int(t) for t in spec.payload.tokens],
        features,  # type: ignore[union-attr]
    )
    assert factor is not None
    bt = compute_target_positions(
        factor,
        min_trade_exposure=float(spec.semantics.neutral_band) or 0.05,
        long_only=not spec.semantics.long_short,
    )[0]
    bars_out = [
        Bar(
            symbol="XAUUSD",
            timeframe=_TF,
            ts=int(panel.time[i]),
            open=float(panel.fields["open"][0, i]),
            high=float(panel.fields["high"][0, i]),
            low=float(panel.fields["low"][0, i]),
            close=float(panel.fields["close"][0, i]),
            volume=float(panel.fields["volume"][0, i]),
            closed=True,
        )
        for i in range(panel.fields["close"].shape[1])
    ]
    rt = list(SignalEngine(spec).replay_closed_bars(bars_out))
    assert rt == pytest.approx([float(p) for p in bt], abs=0.0)
