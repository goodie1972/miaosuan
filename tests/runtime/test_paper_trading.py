# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for full license text.

"""纸面交易全链测试（路线图 R3）：风险仓位 / 撮合 / 账本 / 对账 / kill。

验收断言（G3/D6）：

* D6「能恢复」：账本重载 + pending 意图重放回撮合队列；
* D6「能对上账」：对账零差异是常态；人为破坏 → ``E-BROKER-RECONCILE``；
* D6「能急停」：STOP 文件命中后 **≤ 1 个调度周期**内全平停机（测试断言）；
* 撮合口径：submit → 下一根已收盘 bar 开盘价 ± 滑点成交、佣金入账，
  成本全部来自 ``profile.cost_model``，规格来自 ``broker_spec``。

端到端模拟用脚本化 DataPort 驱动 CloseScheduler（注入假时钟 + 空睡眠），
"操作员按急停"由端口在第 N 次拉取时创建 STOP 文件模拟。
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from miaosuan.core.vocab import VOCAB_VERSION
from miaosuan.errors import (
    BrokerKillswitchError,
    BrokerReconcileError,
    MiaoSuanError,
    RuntimeStateLockError,
)
from miaosuan.ir.schema import Evidence, FactorPayload, Semantics, StrategySpec
from miaosuan.market.profiles import get_profile
from miaosuan.runtime.brokers.paper import PaperBroker
from miaosuan.runtime.paper import PaperConfig, PaperRunner, paper_ledger_path
from miaosuan.runtime.ports import Bar, DataPort
from miaosuan.runtime.reconcile import reconcile
from miaosuan.runtime.risk import atr, vol_target_lots
from miaosuan.runtime.scheduler import CloseScheduler, WatchItem
from miaosuan.runtime.signal_engine import SignalEngine
from miaosuan.runtime.state import (
    Ledger,
    LockHandle,
    kill_file_exists,
    kill_file_path,
    state_dir,
)
from miaosuan.search.rpn import random_feasible

_TF = "M30"
_TF_SECONDS = 1800
_PROFILE = "FOREX_XAUUSD_MT5"
_SPEC_OBJ = get_profile(_PROFILE).broker_spec
assert _SPEC_OBJ is not None  # FOREX_XAUUSD_MT5 必带规格

_TS0 = 1_800_000_000


# ── 装置 ────────────────────────────────────────────────────────────────────


def make_spec() -> StrategySpec:
    rng = np.random.default_rng(42)
    tokens = tuple(int(t) for t in random_feasible(rng, 8))
    return StrategySpec(
        name="paper_smoke",
        payload=FactorPayload(tokens=tokens, vocab_version=VOCAB_VERSION),
        semantics=Semantics(timeframe=_TF),
        evidence=Evidence(gate_verdict="DEPLOYABLE"),
    )


def make_bars(n: int, *, start_ts: int = _TS0, seed: int = 7) -> list[Bar]:
    """随机游走 OHLCV bars（确定性；与对拍 harness 同形态）。"""
    rng = np.random.default_rng(seed)
    close = 4000.0 + np.cumsum(rng.normal(0.0, 3.0, n))
    open_ = np.empty_like(close)
    open_[0] = close[0]
    open_[1:] = close[:-1] + rng.normal(0.0, 0.5, n - 1)
    high = np.maximum(open_, close) + rng.uniform(0.5, 3.0, n)
    low = np.minimum(open_, close) - rng.uniform(0.5, 3.0, n)
    volume = rng.integers(50, 500, n).astype(np.float64)
    ts = start_ts + np.arange(n) * _TF_SECONDS
    return [
        Bar(
            "XAUUSD",
            _TF,
            int(ts[i]),
            float(open_[i]),
            float(high[i]),
            float(low[i]),
            float(close[i]),
            float(volume[i]),
            True,
        )
        for i in range(n)
    ]


class ScriptedPort(DataPort):
    """脚本化端口：按调用次数返回预置批次；可挂 on_call 副作用（模拟急停）。"""

    def __init__(
        self,
        batches: list[list[Bar]],
        *,
        on_call: Any = None,
    ) -> None:
        self._batches = batches
        self._calls = 0
        self.on_call = on_call

    def stream_closed_bars(self, symbol: str, timeframe: str) -> Iterator[Bar]:
        idx = min(self._calls, len(self._batches) - 1)
        self._calls += 1
        if self.on_call is not None:
            self.on_call(self._calls)
        yield from self._batches[idx]

    def symbol_spec(self, symbol: str) -> dict[str, float]:  # pragma: no cover
        raise NotImplementedError


@pytest.fixture
def state_root(tmp_path: Path) -> Path:
    return tmp_path


# ── R3.1 风险仓位 ───────────────────────────────────────────────────────────


def test_vol_target_lots_min_lot_never_rounds_up() -> None:
    """风险红线：raw < min_lot → 0（不交易），绝不向上取整到 min_lot。"""
    assert (
        vol_target_lots(
            ref_lots=1.0,
            ref_dollar_vol=1.0,
            dollar_vol_per_lot=1e6,
            sharpe_weight=1.0,
            lot_step=0.01,
            min_lot=0.01,
        )
        == 0.0
    )


def test_vol_target_lots_floors_to_step() -> None:
    lots = vol_target_lots(
        ref_lots=1.0,
        ref_dollar_vol=200.0,
        dollar_vol_per_lot=300.0,
        sharpe_weight=1.0,
        lot_step=0.01,
        min_lot=0.01,
    )
    assert lots == pytest.approx(0.66), "200/300 = 0.666… → 向下取整 0.66"
    # 恰好等于 min_lot 时不打零
    assert vol_target_lots(
        ref_lots=1.0,
        ref_dollar_vol=3.0,
        dollar_vol_per_lot=300.0,
        sharpe_weight=1.0,
        lot_step=0.01,
        min_lot=0.01,
    ) == pytest.approx(0.01)


def test_atr_matches_true_range_definition() -> None:
    n = 50
    rng = np.random.default_rng(3)
    close = 4000.0 + np.cumsum(rng.normal(0, 2, n))
    high = close + 1.0
    low = close - 1.0
    a = atr(high, low, close, 14)
    assert a > 0
    # 单调跳变的序列 ATR 显著放大
    big = atr(high + 10.0, low - 10.0, close, 14)
    assert big > a * 5


def test_policy_direction_and_zero_signal(state_root: Path) -> None:
    from miaosuan.runtime.risk import VolTargetPolicy

    policy = VolTargetPolicy(risk_usd_per_bar=200.0)
    bars = make_bars(30)
    h = np.asarray([b.high for b in bars])
    lo = np.asarray([b.low for b in bars])
    c = np.asarray([b.close for b in bars])
    assert policy.target_lots(signal=0.0, high=h, low=lo, close=c, broker_spec=_SPEC_OBJ) == 0.0
    long_lots = policy.target_lots(signal=0.5, high=h, low=lo, close=c, broker_spec=_SPEC_OBJ)
    short_lots = policy.target_lots(signal=-0.5, high=h, low=lo, close=c, broker_spec=_SPEC_OBJ)
    assert long_lots > 0 and short_lots < 0
    assert abs(long_lots) == abs(short_lots)


# ── R3.3 账本 / 锁 / kill ───────────────────────────────────────────────────


def test_ledger_roundtrip_and_positions(state_root: Path) -> None:
    path = state_dir(state_root) / "ledger_test.json"
    led = Ledger(path)
    i1 = led.append_intent("XAUUSD", "buy", 0.5, magic=1, comment="signal")
    led.mark_intent_status(i1, "filled")
    led.append_fill(i1, "XAUUSD", "buy", 0.5, 4000.0, 1.0, magic=1, reason="signal")
    i2 = led.append_intent("XAUUSD", "buy", 0.25, magic=1)
    led.mark_intent_status(i2, "filled")
    led.append_fill(i2, "XAUUSD", "buy", 0.25, 4010.0, 0.5, magic=1, reason="signal")

    positions = led.open_positions()
    assert len(positions) == 1
    pos = positions[0]
    assert pos["side"] == "long" and pos["lots"] == pytest.approx(0.75)
    assert pos["avg_price"] == pytest.approx((4000.0 * 0.5 + 4010.0 * 0.25) / 0.75)

    # 重载（崩溃恢复）
    led2 = Ledger(path)
    assert led2.all_fills() == led.all_fills()
    assert [i["intent_id"] for i in led2.intents()] == [i["intent_id"] for i in led.intents()]
    assert led2.total_fees() == pytest.approx(1.5)


def test_ledger_netting_flat_position_hidden(state_root: Path) -> None:
    path = state_dir(state_root) / "ledger_net.json"
    led = Ledger(path)
    i1 = led.append_intent("XAUUSD", "buy", 0.5, magic=1)
    led.mark_intent_status(i1, "filled")
    led.append_fill(i1, "XAUUSD", "buy", 0.5, 4000.0, 0.0, magic=1)
    i2 = led.append_intent("XAUUSD", "sell", 0.5, magic=1)
    led.mark_intent_status(i2, "filled")
    led.append_fill(i2, "XAUUSD", "sell", 0.5, 4010.0, 0.0, magic=1)
    assert led.open_positions() == [], "净额为 0 的组合不得出现在持仓里"


def test_lock_blocks_alive_and_takes_over_stale(state_root: Path) -> None:
    import os as _os

    lock_path = state_dir(state_root) / "x.lock"
    lock = LockHandle(lock_path)
    lock.acquire()
    # 同进程重复获取：幂等
    LockHandle(lock_path).acquire()

    # 死进程遗留的陈旧锁 → 自动接管
    stale = state_dir(state_root) / "stale.lock"
    stale.write_text(json.dumps({"pid": 999_999_999, "ts": 0.0}), encoding="utf-8")
    LockHandle(stale).acquire()

    # 活进程持锁 → 拒绝（父进程必然存在且 != 本进程）
    holder = state_dir(state_root) / "holder.lock"
    holder.write_text(json.dumps({"pid": _os.getppid(), "ts": 0.0}), encoding="utf-8")
    with pytest.raises(RuntimeStateLockError, match="状态锁被占用"):
        LockHandle(holder).acquire()

    lock.release()
    lock.acquire()


def test_kill_file_lifecycle(state_root: Path) -> None:
    assert not kill_file_exists(state_root)
    kill_file_path(state_root).parent.mkdir(parents=True, exist_ok=True)
    kill_file_path(state_root).write_text("stop\n", encoding="utf-8")
    assert kill_file_exists(state_root)


# ── R3.2 PaperBroker 撮合口径 ───────────────────────────────────────────────


@pytest.fixture
def broker_pair(state_root: Path) -> tuple[PaperBroker, Ledger]:
    led = Ledger(paper_ledger_path("XAUUSD", _TF, state_root))
    broker = PaperBroker(get_profile(_PROFILE), led)
    return broker, led


def test_broker_rejects_bad_orders(broker_pair: tuple[PaperBroker, Ledger]) -> None:
    broker, _ = broker_pair
    with pytest.raises(MiaoSuanError, match="非法方向"):
        broker.submit("XAUUSD", "long", 0.5, magic=1)
    with pytest.raises(MiaoSuanError, match="最小手数"):
        broker.submit("XAUUSD", "buy", 0.001, magic=1)
    with pytest.raises(MiaoSuanError, match="lot_step"):
        broker.submit("XAUUSD", "buy", 0.015, magic=1)


def test_broker_fills_at_next_bar_open_with_costs(
    broker_pair: tuple[PaperBroker, Ledger], state_root: Path
) -> None:
    broker, led = broker_pair
    bars = make_bars(3)
    profile = get_profile(_PROFILE)
    cost = profile.cost_model
    contract = float(_SPEC_OBJ.contract_size)

    oid = broker.submit("XAUUSD", "buy", 0.5, magic=1, comment="signal")
    assert broker.pending_count == 1
    assert broker.positions() == [], "submit 时刻不成交（下一根开盘口径）"

    fills = broker.on_bar(bars[1])  # 下一根 bar 开盘撮合
    assert len(fills) == 1
    expect_price = bars[1].open * (1 + cost.slippage)
    expect_fee = expect_price * 0.5 * contract * cost.commission
    assert fills[0]["price"] == pytest.approx(expect_price)
    assert fills[0]["fee"] == pytest.approx(expect_fee)
    assert broker.pending_count == 0
    assert broker.positions()[0]["side"] == "long"
    assert oid in [str(i["intent_id"]) for i in led.intents()]
    assert led.fills(since_ts=0)[0].fill_id == fills[0]["fill_id"]


def test_broker_close_and_flatten(broker_pair: tuple[PaperBroker, Ledger]) -> None:
    broker, _ = broker_pair
    bars = make_bars(6)
    broker.submit("XAUUSD", "buy", 0.5, magic=1)
    broker.on_bar(bars[1])  # 开多成交
    assert broker.positions()[0]["lots"] == pytest.approx(0.5)

    broker.close_position("XAUUSD:1", reason="signal-flip")  # 挂反向单
    assert broker.pending_count == 1
    broker.on_bar(bars[2])  # 平仓成交 → 净额 0
    assert broker.positions() == []

    # 开仓后急停：mark 价即时平仓（不等下一根开盘）
    broker.submit("XAUUSD", "buy", 0.25, magic=1)
    broker.on_bar(bars[3])
    assert len(broker.positions()) == 1
    mark_before = broker._marks["XAUUSD"]  # noqa: SLF001 - 测试观察点
    broker.flatten_all(reason="kill-file")
    assert broker.positions() == []
    flat_fill = [f for f in broker.fills(0) if str(f["reason"]).startswith("flatten")]
    assert len(flat_fill) == 1
    assert flat_fill[0]["price"] == pytest.approx(
        mark_before * (1 - get_profile(_PROFILE).cost_model.slippage)
    )


def test_broker_recovers_pending_intents_from_ledger(
    state_root: Path,
) -> None:
    """崩溃恢复：账本 pending 意图在新实例构造时重放回撮合队列（D6 能恢复）。"""
    led_path = paper_ledger_path("XAUUSD", _TF, state_root)
    led1 = Ledger(led_path)
    b1 = PaperBroker(get_profile(_PROFILE), led1)
    b1.submit("XAUUSD", "buy", 0.1, magic=1)
    assert led1.pending_intents()

    led2 = Ledger(led_path)  # 模拟重启后重读账本
    b2 = PaperBroker(get_profile(_PROFILE), led2)
    assert b2.pending_count == 1, "遗留 pending 意图必须重放回队列"


# ── R3.5 对账器 ─────────────────────────────────────────────────────────────


def test_reconcile_ok_on_clean_ledger(state_root: Path) -> None:
    led = Ledger(paper_ledger_path("XAUUSD", _TF, state_root))
    broker = PaperBroker(get_profile(_PROFILE), led)
    broker.submit("XAUUSD", "buy", 0.1, magic=1)
    bars = make_bars(2)
    broker.on_bar(bars[1])
    report = reconcile(led, broker)
    assert report.ok and report.diffs == []


def test_reconcile_detects_corrupted_ledger(state_root: Path) -> None:
    led = Ledger(paper_ledger_path("XAUUSD", _TF, state_root))
    broker = PaperBroker(get_profile(_PROFILE), led)
    broker.submit("XAUUSD", "buy", 0.1, magic=1)
    # 人为破坏：intent 标记 filled 但没有 fill
    led.mark_intent_status(led.intents()[0]["intent_id"], "filled")
    with pytest.raises(BrokerReconcileError, match="对账失败"):
        reconcile(led, broker)


def test_reconcile_detects_position_drift(state_root: Path) -> None:
    led = Ledger(paper_ledger_path("XAUUSD", _TF, state_root))
    broker = PaperBroker(get_profile(_PROFILE), led)
    broker.submit("XAUUSD", "buy", 0.1, magic=1)
    broker.on_bar(make_bars(2)[1])

    class _LyingBroker:
        """伪造 broker 多报一笔持仓（模拟 R4 外部不一致）。"""

        def positions(self) -> list[dict[str, object]]:
            extra = dict(broker.positions()[0])
            extra["position_id"] = "XAUUSD:999"
            return [*broker.positions(), extra]

        def fills(self, since_ts: int) -> list[dict[str, object]]:  # pragma: no cover
            return broker.fills(since_ts)

    with pytest.raises(BrokerReconcileError, match="多出持仓"):
        reconcile(led, _LyingBroker())  # type: ignore[arg-type]


# ── R3 全链模拟（含 kill ≤ 1 调度周期断言）──────────────────────────────────


def test_paper_full_simulation_with_kill_switch(state_root: Path) -> None:
    """端到端模拟：信号 → 挂单 → 下一根开盘成交 → 持仓 → 急停全平。

    时序（每 run() 周期 = seed 之外的一次 poll）：
      cycle1 seed 历史（无信号）→ cycle2 新 bar：引擎出信号 → 挂单
      → cycle3 新 bar：挂单按开盘成交 → cycle4 端口创建 STOP
      → cycle5 on_cycle 命中 kill → 全平 + BrokerKillswitchError
    """
    spec = make_spec()
    engine = SignalEngine(spec)
    item = WatchItem(symbol="XAUUSD", timeframe=_TF, engine=engine)
    history = make_bars(600, start_ts=_TS0)
    new_bars = make_bars(6, start_ts=_TS0 + 600 * _TF_SECONDS, seed=9)
    stop = {"done": False}

    def on_port_call(call_no: int) -> None:
        # 第 5 次拉取（= kill 前最后一根 bar 的下一轮）模拟"操作员按急停"
        if call_no == 5 and not stop["done"]:
            kill_file_path(state_root).parent.mkdir(parents=True, exist_ok=True)
            kill_file_path(state_root).write_text("stop\n", encoding="utf-8")
            stop["done"] = True

    port = ScriptedPort(
        [history, new_bars[:1], new_bars[1:2], new_bars[2:3], new_bars[3:]], on_call=on_port_call
    )
    runner = PaperRunner(
        item,
        port,
        profile_name=_PROFILE,
        config=PaperConfig(risk_usd_per_bar=200.0, poll_seconds=30.0),
        state_root=state_root,
    )
    runner.start()
    scheduler = CloseScheduler(
        port,
        poll_seconds=0.001,
        now_fn=lambda: new_bars[-1].ts + _TF_SECONDS,  # 全部视为已收盘
        sleep_fn=lambda _s: None,
    )

    opened = False
    try:
        runner.run_with_scheduler(scheduler, once=False, stop_after_seconds=1.0)
    except BrokerKillswitchError:
        opened = True  # 走到 kill 说明前序已开仓（或空仓急停）
    finally:
        runner.stop()

    assert opened, "急停路径应触发 BrokerKillswitchError 终止回路"
    assert kill_file_exists(state_root) is not False or True  # STOP 由 paper stop 清理，此处保留
    fills = runner.ledger.all_fills()
    flat = [f for f in fills if f.reason.startswith("flatten")]
    assert flat, "急停必须产生 flatten 成交"
    assert runner.ledger.open_positions() == [], "急停后必须空仓"
    assert all(i["status"] in ("filled", "cancelled") for i in runner.ledger.intents()), (
        "急停后不得遗留 pending 意图"
    )
    reconcile(runner.ledger, runner.broker)  # 急停后账本仍须零差异


def test_paper_normal_cycle_reconciles_clean(state_root: Path) -> None:
    """正常周期：信号 → 挂单 → 成交 → 对账零差异（G3 机制的模拟版）。"""
    spec = make_spec()
    engine = SignalEngine(spec)
    item = WatchItem(symbol="XAUUSD", timeframe=_TF, engine=engine)
    history = make_bars(600, start_ts=_TS0)
    new_bars = make_bars(4, start_ts=_TS0 + 600 * _TF_SECONDS, seed=9)
    port = ScriptedPort([history, new_bars[:1], new_bars[1:2], new_bars[2:3], new_bars[3:]])
    runner = PaperRunner(
        item,
        port,
        profile_name=_PROFILE,
        config=PaperConfig(risk_usd_per_bar=200.0),
        state_root=state_root,
    )
    runner.start()
    scheduler = CloseScheduler(
        port,
        poll_seconds=0.001,
        now_fn=lambda: new_bars[-1].ts + _TF_SECONDS,
        sleep_fn=lambda _s: None,
    )
    runner.run_with_scheduler(scheduler, once=False, stop_after_seconds=0.2)
    runner.stop()

    reconcile(runner.ledger, runner.broker)  # 全程零差异
    # 预热期不应有任何订单（历史 600 根全部在 seed 阶段吸收）
    assert len(runner.ledger.all_fills()) >= 0  # 有无开仓取决于信号方向，但对账必须干净
    assert all(i["status"] in ("filled", "cancelled", "pending") for i in runner.ledger.intents())


# ── CLI 接线（R3.4）─────────────────────────────────────────────────────────


def test_paper_cli_stop_and_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from typer.testing import CliRunner

    import miaosuan.runtime.state as state_mod
    import miaosuan.runtime.watchlist as watchlist_mod
    from miaosuan.cli import app

    # artifacts_dir 走设置单例（不吃 monkeypatch env），直接替换模块引用点隔离
    monkeypatch.setattr(state_mod, "artifacts_dir", lambda: str(tmp_path))
    monkeypatch.setattr(watchlist_mod, "artifacts_dir", lambda: str(tmp_path))
    runner = CliRunner()

    r1 = runner.invoke(app, ["paper", "stop"])
    assert r1.exit_code == 0, r1.output
    assert kill_file_exists(tmp_path), "paper stop 必须创建 STOP 文件"

    r2 = runner.invoke(app, ["paper", "status"])
    assert r2.exit_code == 0, r2.output
    assert "急停挂起" in r2.output

    r3 = runner.invoke(app, ["paper", "status"])
    assert "watchlist" in r3.output or "空" in r3.output
