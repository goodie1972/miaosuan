# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for full license text.

"""runtime 单测：信号引擎（去重/收盘纪律/安全链）+ 调度器 + watchlist + 数据端口。

对拍（G2）另见 ``test_runtime_signal_parity.py``；本文件只测运行时自身约定。
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from miaosuan.core.vocab import VOCAB_VERSION
from miaosuan.errors import (
    MiaoSuanError,
    RuntimeNotDeployableError,
    RuntimeSignalError,
    VocabVersionMismatchError,
)
from miaosuan.ir.codec import write_spec
from miaosuan.ir.schema import (
    Evidence,
    FactorPayload,
    ParamPayload,
    Semantics,
    StrategySpec,
)
from miaosuan.runtime import SignalEngine, load_spec_checked
from miaosuan.runtime.ports import Bar, DataPort
from miaosuan.runtime.scheduler import (
    CloseScheduler,
    WatchItem,
    log_bar_event,
    timeframe_seconds,
)
from miaosuan.runtime.watchlist import WatchEntry, WatchlistStore
from miaosuan.search.rpn import random_feasible

_TF = "M30"


def make_spec(**sem_kwargs: Any) -> StrategySpec:
    rng = np.random.default_rng(42)
    tokens = tuple(int(t) for t in random_feasible(rng, 8))
    return StrategySpec(
        name="engine_smoke",
        payload=FactorPayload(tokens=tokens, vocab_version=VOCAB_VERSION),
        semantics=Semantics(timeframe=_TF, **sem_kwargs),
        evidence=Evidence(gate_verdict="DEPLOYABLE"),
    )


def make_bars(n: int, *, start_ts: int = 1_800_000_000, closed: bool = True) -> list[Bar]:
    """确定性上升序列（简单起见 close 递增，形状满足 OHLC 约束）。"""
    out: list[Bar] = []
    price = 4000.0
    for i in range(n):
        price += 0.5
        out.append(
            Bar(
                "XAUUSD",
                _TF,
                start_ts + i * 1800,
                price - 0.2,
                price + 1.0,
                price - 1.0,
                price,
                100.0,
                closed,
            )
        )
    return out


class FakeDataPort(DataPort):
    """脚本化 DataPort：按调用次数返回预置批次（水位内部维护）。"""

    def __init__(self, batches: list[list[Bar]]) -> None:
        self._batches = batches
        self._calls = 0

    def stream_closed_bars(self, symbol: str, timeframe: str) -> Iterator[Bar]:
        idx = min(self._calls, len(self._batches) - 1)
        self._calls += 1
        yield from self._batches[idx]

    def symbol_spec(self, symbol: str) -> dict[str, float]:  # pragma: no cover
        raise NotImplementedError


# ── 信号引擎 ────────────────────────────────────────────────────────────────


def test_engine_rejects_unclosed_bar() -> None:
    engine = SignalEngine(make_spec())
    with pytest.raises(RuntimeSignalError, match="收盘纪律"):
        engine.on_closed_bar(make_bars(1, closed=False)[0])


def test_engine_dedups_repeated_bars() -> None:
    spec = make_spec()
    bars = make_bars(600)
    engine = SignalEngine(spec)
    first = engine.on_closed_bar(bars[-1])  # 跳过预热直接喂末根
    again = engine.on_closed_bar(bars[-1])
    assert first == again
    # 回退时间戳同样幂等
    older = engine.on_closed_bar(bars[-2])
    assert older == first


def test_engine_warmup_returns_zero() -> None:
    spec = make_spec()  # warmup_bars=499
    engine = SignalEngine(spec)
    for bar in make_bars(400):
        engine.on_closed_bar(bar)
    assert engine.last_position == 0.0, "预热期（< warmup+1 根）必须出 0"
    for bar in make_bars(150, start_ts=1_800_000_000 + 400 * 1800):
        engine.on_closed_bar(bar)
    assert engine.spec is spec


def test_engine_rejects_non_factor_payload() -> None:
    spec = StrategySpec(
        name="tpl",
        payload=ParamPayload(template_id="t", params={}),
        semantics=Semantics(timeframe=_TF),
    )
    with pytest.raises(RuntimeSignalError, match="因子载荷"):
        SignalEngine(spec)


def test_engine_rejects_unknown_position_fn() -> None:
    spec = StrategySpec(
        name="bad",
        payload=FactorPayload(tokens=(33, 62), vocab_version=VOCAB_VERSION),
        semantics=Semantics(position_fn="sigmoid"),
    )
    with pytest.raises(RuntimeSignalError, match="仓位映射"):
        SignalEngine(spec)


# ── 启动安全链（Sample G）───────────────────────────────────────────────────


def test_load_spec_rejects_non_deployable(tmp_path: Path) -> None:
    spec = make_spec()
    bad = StrategySpec(
        name=spec.name,
        payload=spec.payload,
        semantics=spec.semantics,
        evidence=Evidence(gate_verdict="RESEARCH_ONLY"),
    )
    path = tmp_path / "bad.json"
    write_spec(path, bad)
    with pytest.raises(RuntimeNotDeployableError):
        load_spec_checked(path)


def test_load_spec_rejects_vocab_mismatch(tmp_path: Path) -> None:
    spec = make_spec()
    bad = StrategySpec(
        name=spec.name,
        payload=FactorPayload(tokens=spec.payload.tokens, vocab_version="v_deadmoo"),  # type: ignore[union-attr]
        semantics=spec.semantics,
        evidence=Evidence(gate_verdict="DEPLOYABLE"),
    )
    path = tmp_path / "bad_vocab.json"
    write_spec(path, bad)
    with pytest.raises(VocabVersionMismatchError):
        load_spec_checked(path)


def test_load_spec_accepts_deployable(tmp_path: Path) -> None:
    path = tmp_path / "ok.json"
    write_spec(path, make_spec())
    loaded = load_spec_checked(path)
    assert loaded.evidence.gate_verdict == "DEPLOYABLE"


# ── 调度器 ──────────────────────────────────────────────────────────────────


def test_timeframe_seconds_unknown_raises() -> None:
    with pytest.raises(MiaoSuanError):
        timeframe_seconds("H7")


def test_scheduler_seed_does_not_emit_signals() -> None:
    spec = make_spec()
    bars = make_bars(600)
    port = FakeDataPort([bars])
    item = WatchItem(symbol="XAUUSD", timeframe=_TF, engine=SignalEngine(spec))
    scheduler = CloseScheduler(port)

    last = scheduler.seed_watermark(item, now_utc=bars[-1].ts + 1800)

    assert last == bars[-1].ts
    assert item.engine.bars_seen == 502, "只回填尾部 warmup+缓冲 根（防 O(T²)）"
    assert scheduler.poll(item, now_utc=bars[-1].ts + 1800) == [], "历史不应作为新收盘再发"


def test_scheduler_emits_only_new_closed_bars() -> None:
    spec = make_spec()
    bars = make_bars(600)
    newer = make_bars(2, start_ts=bars[-1].ts + 1800)
    # 第一批=历史（seed），第二批=新收盘
    port = FakeDataPort([bars, newer, newer])
    item = WatchItem(symbol="XAUUSD", timeframe=_TF, engine=SignalEngine(spec))
    scheduler = CloseScheduler(port)
    scheduler.seed_watermark(item, now_utc=bars[-1].ts + 1800)

    fresh = scheduler.poll(item, now_utc=newer[-1].ts + 1800)
    assert [b.ts for b in fresh] == [b.ts for b in newer], "只发新增收盘 bar"
    # 重复轮询：无新 bar（去重）
    assert scheduler.poll(item, now_utc=newer[-1].ts + 1800) == []


def test_scheduler_filters_forming_bar() -> None:
    """未走完周期的 bar（ts+tf > now）即使端口泄漏也不得下发。"""
    spec = make_spec()
    bars = make_bars(600)
    b = bars[-1]
    forming = Bar(
        b.symbol, b.timeframe, b.ts + 1800, b.open, b.high, b.low, b.close, b.volume, True
    )
    port = FakeDataPort([[*bars, forming]])
    item = WatchItem(symbol="XAUUSD", timeframe=_TF, engine=SignalEngine(spec))
    scheduler = CloseScheduler(port)
    scheduler.seed_watermark(item, now_utc=bars[-1].ts + 1800)
    assert scheduler.poll(item, now_utc=bars[-1].ts + 1800) == []


# ── 转折通知 ────────────────────────────────────────────────────────────────


def test_log_bar_event_notifies_on_direction_turn() -> None:
    spec = make_spec()
    bars = make_bars(600)
    item = WatchItem(symbol="XAUUSD", timeframe=_TF, engine=SignalEngine(spec))
    item.engine.on_closed_bar(bars[-1])
    notified: list[tuple[str, str]] = []

    class _N:
        def notify(self, title: str, body: str, *, level: str = "info") -> None:
            notified.append((title, body))

    pos = item.engine.last_position
    # 同号：不通知
    log_bar_event(item, bars[-1], pos, pos, _N())  # type: ignore[arg-type]
    assert notified == []
    # 反号：通知
    log_bar_event(item, bars[-1], -pos if pos != 0 else 1.0, pos, _N())  # type: ignore[arg-type]
    assert len(notified) == 1


# ── watchlist 存储 ──────────────────────────────────────────────────────────


def test_watchlist_store_roundtrip(tmp_path: Path) -> None:
    store = WatchlistStore(tmp_path / "watchlist.json")
    assert store.load() == []
    store.add(WatchEntry(symbol="XAUUSD", spec_path="artifacts/a.json", profile="FOREX_XAUUSD_MT5"))
    store.add(WatchEntry(symbol="BTCUSDT", spec_path="artifacts/b.json"))
    store.add(WatchEntry(symbol="XAUUSD", spec_path="artifacts/c.json"))  # 同品种覆盖
    entries = store.load()
    assert [e.symbol for e in entries] == ["BTCUSDT", "XAUUSD"]
    assert entries[1].spec_path == "artifacts/c.json"
    assert store.remove("BTCUSDT") is True
    assert store.remove("NOPE") is False
    assert [e.symbol for e in store.load()] == ["XAUUSD"]
    raw = json.loads((tmp_path / "watchlist.json").read_text(encoding="utf-8"))
    assert set(raw["entries"][0]) == {"symbol", "spec_path", "profile", "timeframe"}


def test_watchlist_corrupt_file_raises(tmp_path: Path) -> None:
    p = tmp_path / "watchlist.json"
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(MiaoSuanError, match="watchlist 损坏"):
        WatchlistStore(p).load()


# ── MT5DataPort ─────────────────────────────────────────────────────────────


class _StubFetcher:
    """脚本化 fetcher：返回预置 DataFrame，记录 since_ts。"""

    def __init__(self, frames: list[pd.DataFrame]) -> None:
        self._frames = frames
        self._calls = 0
        self.since_calls: list[int] = []

    def fetch_full(self, symbol: str, timeframe: str) -> pd.DataFrame:
        self._calls += 1
        return self._frames[min(self._calls, len(self._frames)) - 1]

    def fetch_incremental(self, symbol: str, timeframe: str, since_ts: int) -> pd.DataFrame:
        self.since_calls.append(since_ts)
        self._calls += 1
        return self._frames[min(self._calls, len(self._frames)) - 1]


def _frame(ts_list: list[int]) -> pd.DataFrame:
    n = len(ts_list)
    return pd.DataFrame(
        {
            "time": ts_list,
            "open": [4000.0 + i for i in range(n)],
            "high": [4001.0 + i for i in range(n)],
            "low": [3999.0 + i for i in range(n)],
            "close": [4000.5 + i for i in range(n)],
            "volume": [10.0] * n,
            "tick_volume": [10] * n,
        }
    )


def test_mt5_dataport_filters_forming_and_dedups() -> None:
    from miaosuan.runtime.sources.mt5 import MT5DataPort

    tf = timeframe_seconds(_TF)
    ts_first = 1_800_000_000
    closed_ts = [ts_first + i * tf for i in range(10)]  # 已收盘 10 根
    forming_ts = ts_first + 10 * tf  # forming bar
    now_before = forming_ts + tf - 1  # forming 未走完
    now_after = forming_ts + tf  # 该周期走完 → 变成已收盘

    fetcher = _StubFetcher(
        [
            _frame(closed_ts + [forming_ts]),  # 首轮 full
            _frame(closed_ts + [forming_ts]),  # incremental（无新）
            _frame(closed_ts + [forming_ts, forming_ts + tf]),  # incremental（多一根新收盘）
        ]
    )
    port = MT5DataPort(fetcher, now_fn=lambda: now_before)  # type: ignore[arg-type]

    first = list(port.stream_closed_bars("XAUUSD", _TF))
    assert len(first) == 10, "forming bar 不得产出"
    assert all(b.closed for b in first)

    # 无新收盘 → 空（水位去重）
    assert list(port.stream_closed_bars("XAUUSD", _TF)) == []

    # 时钟前进：原 forming 周期走完，成为第 11 根已收盘
    port._now_fn = lambda: now_after
    grown = list(port.stream_closed_bars("XAUUSD", _TF))
    assert [b.ts for b in grown] == [forming_ts]
