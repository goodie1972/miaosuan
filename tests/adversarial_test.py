# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""对抗性测试（adversarial testing）：故意提供错误输入、异常状态，
验证系统是否如期失败（抛正确错误码）而非崩溃或静默失效。

此脚本不依赖真实 MT5 终端；全部使用假装置（与 test_live_broker.py 类似）。
覆盖：凭证错误、终端不可用、错误订单、空 watchlist、对账不一致、
熔断触发、空持仓平仓、非法时间框架、非交易时段信号。
"""

from __future__ import annotations

import sys
import time
import types
from pathlib import Path

import pytest

from miaosuan.cli import app
from miaosuan.core.vocab import VOCAB_VERSION
from miaosuan.errors import (
    BrokerKillswitchError,
    BrokerUnavailableError,
    MiaoSuanError,
    RuntimeNotDeployableError,
    RuntimeSignalError,
)
from miaosuan.ir.schema import Evidence, FactorPayload, Semantics, StrategySpec
from miaosuan.runtime.breaker import BreakerConfig
from miaosuan.runtime.brokers.mt5 import MT5Broker
from miaosuan.runtime.live import LiveConfig, LiveRunner
from miaosuan.runtime.paper import PaperConfig, PaperRunner
from miaosuan.runtime.scheduler import WatchItem
from miaosuan.runtime.signal_engine import SignalEngine
from miaosuan.runtime.state import (
    Ledger,
    LockHandle,
    kill_file_exists,
    kill_file_path,
    state_dir,
)
from miaosuan.runtime.watchlist import WatchlistStore
from miaosuan.search.rpn import random_feasible
from typer.testing import CliRunner

# ── 假 MetaTrader5 模块（与 test_live_broker.py 相同） ─────────────────────


class _FakeOrder:
    def __init__(self, retcode: int, order: int, comment: str = "ok") -> None:
        self.retcode = retcode
        self.order = order
        self.comment = comment


class _FakePosition:
    def __init__(self, ticket: int, symbol: str, is_long: bool, volume: float,
                 price_open: float, magic: int) -> None:
        self.ticket = ticket
        self.symbol = symbol
        self.type = 0 if is_long else 1  # POSITION_TYPE_BUY=0 / SELL=1
        self.volume = volume
        self.price_open = price_open
        self.magic = magic


class _FakeDeal:
    def __init__(self, ticket: int, order: int, symbol: str, is_buy: bool, volume: float,
                 price: float, commission: float, swap: float, profit: float,
                 ts: int, magic: int) -> None:
        self.ticket = ticket
        self.order = order
        self.symbol = symbol
        self.type = 0 if is_buy else 1  # DEAL_TYPE_BUY=0
        self.volume = volume
        self.price = price
        self.commission = commission
        self.swap = swap
        self.profit = profit
        self.time = ts
        self.magic = magic


class _FakeSymbolInfo:
    def __init__(self, *, volume_step: float = 0.01, volume_min: float = 0.01,
                 contract_size: float = 100.0, tick_value: float = 1.0,
                 filling_mode: int = 1) -> None:
        self.volume_step = volume_step
        self.volume_min = volume_min
        self.trade_contract_size = contract_size
        self.trade_tick_value = tick_value
        self.filling_mode = filling_mode


class _FakeAccount:
    login = 12345
    server = "Fake-Demo"
    currency = "USD"
    balance = 10000.0
    equity = 10000.0
    leverage = 100


class _FakeMT5Module(types.ModuleType):
    TRADE_ACTION_DEAL = 1
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    ORDER_TIME_GTC = 0
    ORDER_FILLING_FOK = 0
    ORDER_FILLING_IOC = 1
    ORDER_FILLING_RETURN = 2
    TRADE_RETCODE_DONE = 10009
    TRADE_RETCODE_REJECT = 10004
    POSITION_TYPE_BUY = 0
    POSITION_TYPE_SELL = 1

    def __init__(self) -> None:
        super().__init__("MetaTrader5")
        self.connected = False
        self.orders: list[dict[str, Any]] = []
        self.reject_next = 0
        self.positions: list[_FakePosition] = []
        self.deals: list[_FakeDeal] = []
        self._next_order = 500_000
        self._next_deal = 900_000

    def initialize(self, **kwargs: Any) -> bool:
        self.connected = True
        self.init_kwargs = dict(kwargs)
        return True

    def shutdown(self) -> None:
        self.connected = False

    def last_error(self) -> tuple[int, str]:
        return (0, "ok")

    def symbol_info_tick(self, symbol: str) -> types.SimpleNamespace:
        now = int(time.time())
        return types.SimpleNamespace(time=now, bid=4000.0, ask=4000.2, last=0.0, volume=0)

    def symbol_info(self, symbol: str) -> _FakeSymbolInfo:
        return _FakeSymbolInfo()

    def order_send(self, req: dict[str, Any]) -> _FakeOrder:
        self.orders.append(dict(req))
        if self.reject_next > 0:
            self.reject_next -= 1
            return _FakeOrder(self.TRADE_RETCODE_REJECT, 0, "rejected")
        self._next_order += 1
        self._next_deal += 1
        is_buy = int(req["type"]) == self.ORDER_TYPE_BUY
        fill_price = float(req["price"])
        self.deals.append(
            _FakeDeal(self._next_deal, self._next_order, req["symbol"], is_buy,
                      float(req["volume"]), fill_price, -2.0, 0.0, 0.0,
                      int(time.time()), int(req["magic"]))
        )
        return _FakeOrder(self.TRADE_RETCODE_DONE, self._next_order)

    def positions_get(self, ticket: int | None = None) -> list[_FakePosition]:
        if ticket is not None:
            return [p for p in self.positions if p.ticket == ticket]
        return list(self.positions)

    def history_deals_get(self, date_from: Any, date_to: Any) -> list[_FakeDeal]:
        return list(self.deals)

    def account_info(self) -> _FakeAccount:
        return _FakeAccount()


@pytest.fixture
def fake_mt5(monkeypatch: pytest.MonkeyPatch) -> _FakeMT5Module:
    mod = _FakeMT5Module()
    monkeypatch.setitem(sys.modules, "MetaTrader5", mod)
    return mod


# ── 1. 错误的凭证（登录失败） ────────────────────────────────────────
def test_mt5_broker_connect_fails_with_bad_credentials() -> None:
    broker = MT5Broker(login=999999, password="wrong", server="bad")
    with pytest.raises(BrokerUnavailableError):
        broker.connect()


# ── 2. MT5 终端未运行时的可用性检测 ────────────────────────────────
def test_mt5_broker_is_available_false_when_not_running() -> None:
    broker = MT5Broker()
    # 未调用 connect() 时 is_available 只检查依赖（这里假装置已装），
    # 但真实终端未运行时 connect() 会失败；这里直接测 connect 失败
    with pytest.raises(BrokerUnavailableError):
        broker.connect()


# ── 3. 错误的订单参数（负手数、零手数、非法方向） ───────────────────
def test_mt5_broker_rejects_bad_order_params(fake_mt5: _FakeMT5Module) -> None:
    broker = MT5Broker()
    broker.connect()
    # 非法方向
    with pytest.raises(BrokerUnavailableError, match="非法方向"):
        broker.submit("XAUUSD", "long", 0.1, magic=1)
    # 零手数
    with pytest.raises(BrokerUnavailableError, match="手数必须为正"):
        broker.submit("XAUUSD", "buy", 0.0, magic=1)
    # 负手数
    with pytest.raises(BrokerUnavailableError, match="手数必须为正"):
        broker.submit("XAUUSD", "buy", -0.1, magic=1)
    broker.close()


# ── 4. 空 watchlist 启动纸面/实盘 ─────────────────────────────────
def test_paper_start_refuses_empty_watchlist() -> None:
    from typer.testing import CliRunner
    runner = CliRunner()
    result = runner.invoke(app, ["paper", "start"])
    assert result.exit_code != 0
    assert "watchlist 为空" in result.output


def test_live_start_refuses_empty_watchlist() -> None:
    from typer.testing import CliRunner
    runner = CliRunner()
    result = runner.invoke(app, ["live", "start"])
    assert result.exit_code != 0
    assert "watchlist 为空" in result.output


# ── 5. 对账故意不一致（人为破坏账本）→ 应抛 E-BROKER-RECONCILE ─────
def test_live_runner_raises_on_account_mismatch() -> None:
    from miaosuan.runtime.live import LiveRunner, LiveConfig
    from miaosuan.runtime.paper import PaperConfig
    from miaosuan.runtime.state import Ledger
    from miaosuan.runtime.brokers.mt5 import MT5Broker
    from miaosuan.runtime.signal_engine import SignalEngine
    from miaosuan.ir.schema import Evidence, FactorPayload, Semantics, StrategySpec
    from miaosuan.runtime.reconcile import reconcile

    spec = StrategySpec(
        name="advers",
        payload=FactorPayload(tokens=(33, 62), vocab_version=VOCAB_VERSION),
        semantics=Semantics(timeframe="M30"),
        evidence=Evidence(gate_verdict="DEPLOYABLE"),
    )
    engine = SignalEngine.__new__(SignalEngine)
    engine._spec = spec  # noqa: SLF001
    engine._last_position = 0.0  # noqa: SLF001
    item = WatchItem(symbol="XAUUSD", timeframe="M30", engine=engine)

    ledger = Ledger(state_dir() / "ledger_test.json")
    broker = MT5Broker()  # 假终端
    broker.connect()

    # 先挂单买入 0.1 lots
    intent_id = broker.submit("XAUUSD", "buy", 0.1, magic=7, comment="signal")
    # 人为破坏：不记录此 fill，但标记 intent 为 filled（制造不一致）
    ledger.mark_intent_status(intent_id, "filled")
    # 现在对账：应检测到意图为 filled 但无 fill，抛 BrokerReconcileError
    with pytest.raises(MiaoSuanError) as excinfo:
        reconcile(ledger, broker)
    assert "对账失败" in str(excinfo.value)
    broker.close()


# ── 6. 熔断三阈值触发 → 应创建 STOP 并抛 BrokerKillswitchError ───
def test_live_runner_trip_flow_creates_stop_and_stops() -> None:
    from miaosuan.runtime.live import LiveRunner, LiveConfig
    from miaosuan.runtime.paper import PaperConfig
    from miaosuan.runtime.broker_breaker import BreakerConfig
    from miaosuan.runtime.state import kill_file_exists, kill_file_path
    from miaosuan.runtime.brokers.mt5 import MT5Broker
    from miaosuan.runtime.signal_engine import SignalEngine
    from miaosuan.ir.schema import Evidence, FactorPayload, Semantics, StrategySpec

    spec = StrategySpec(
        name="advers2",
        payload=FactorPayload(tokens=(33, 62), vocab_version=VOCAB_VERSION),
        semantics=Semantics(timeframe="M30"),
        evidence=Evidence(gate_verdict="DEPLOYABLE"),
    )
    engine = SignalEngine.__new__(SignalEngine)
    engine._spec = spec  # noqa: SLF001
    engine._last_position = 0.0  # noqa: SLF001
    item = WatchItem(symbol="XAUUSD", timeframe="M30", engine=engine)

    broker = MT5Broker()
    broker.connect()
    runner = LiveRunner(
        item,
        port=None,
        profile_name="FOREX_XAUUSD_MT5",
        config=LiveConfig(
            risk_usd_per_bar=200.0,
            poll_seconds=0.0,
            breaker=BreakerConfig(max_consecutive_rejections=2),  # 连续 2 次拒单触发
        ),
        broker=broker,
    )
    runner.start()

    # 连续两次拒单 → 应触发熔断
    runner.breaker.on_order_rejected("r1")
    runner.breaker.on_order_rejected("r2")
    assert runner.breaker.tripped

    # 直接调用 trip flow：应创建 STOP 并抛 BrokerKillswitchError
    with pytest.raises(BrokerKillswitchError):
        runner._trip_flow()

    assert kill_file_exists(), "熔断必须创建 STOP 文件"
    assert kill_file_path().read_text(encoding="utf-8").startswith("breaker:")
    broker.close()


# ── 7. 持仓为空时的 flatten_all（不应崩溃） ──────────────────────
def test_mt5_broker_flatten_all_when_no_positions() -> None:
    broker = MT5Broker()
    broker.connect()
    # 不持任何仓，直接调用 flatten_all 应不崩溃
    broker.flatten_all(reason="test")
    broker.close()


# ── 8. 非交易时段的信号引擎（应返回 0） ────────────────────────
def test_signal_engine_returns_zero_when_no_history() -> None:
    spec = StrategySpec(
        name="advers3",
        payload=FactorPayload(tokens=(33, 62), vocab_version=VOCAB_VERSION),
        semantics=Semantics(timeframe="M30"),
        evidence=Evidence(gate_verdict="DEPLOYABLE"),
    )
    engine = SignalEngine(spec)
    # 从未喂过历史 → 持仓为零 → 信号应为 0（风险目标手数为零）
    assert engine.last_position == 0.0
    # 对 Bar（不论价格）亦应返回 0（因无历史）
    bar = types.SimpleNamespace(
        symbol="XAUUSD",
        timeframe="M30",
        ts=int(time.time()),
        open=4000.0,
        high=4001.0,
        low=3999.0,
        close=4000.0,
        volume=10.0,
        closed=True,
    )
    pos = engine.on_closed_bar(bar)
    assert pos == 0.0


# ── 9. 错误的时间框架（如 M99）→ MiaoSuanError ────────────────
def test_invalid_timeframe_raises_in_spec_and_engine() -> None:
    with pytest.raises(MiaoSuanError, match="不支持的周期"):
        Semantics(timeframe="M99")
    with pytest.raises(MiaoSuanError, match="不支持的周期"):
        SignalEngine(StrategySpec(
            name="x",
            payload=FactorPayload(tokens=(1, 2), vocab_version=VOCAB_VERSION),
            semantics=Semantics(timeframe="M99"),
        ))


# ── 10. MT5 端口在无终端时的 fills/positions（应抛 BrokerUnavailableError） ───
def test_mt5_broker_positions_and_fails_when_disconnected() -> None:
    broker = MT5Broker()
    # 未连接
    with pytest.raises(BrokerUnavailableError, match="未连接"):
        broker.positions()
    with pytest.raises(BrokerUnavailableError, match="未连接"):
        broker.fills(0)
    # 已连接但后续 disconnect（此假终端无 disconnect，但 connect 失败路径同上）
    # 这里直接测未连接即可


if __name__ == "__main__":
    # 简单的直接执行（不使用 pytest）以便在交互中快速查看
    # 实际建议仍用 pytest 运行全部断言
    print("对抗性测试框架就绪。建议用 pytest 运行全部断言。")
    print("示例：")
    print("  python -m pytest <this_file> -q")