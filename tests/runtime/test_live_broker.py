# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for full license text.

"""实盘执行测试（路线图 R4）：假 MetaTrader5 模块全覆盖，**绝不下真实订单**。

覆盖：DEAL 下单（price/deviation/type_time/magic）/ retcode 拒单 /
ticket 平仓（hedging ``position`` 字段）/ 持仓与回执映射 / 品种规格 /
熔断三阈值（连续拒单/日亏/滑点）/ magic 域隔离 / 真钱 CLI 门禁。
真实终端只做只读冒烟（见体检流程），订单路径的实盘验证在 R4.5 模拟户
手工清单中执行。
"""

from __future__ import annotations

import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from typer.testing import CliRunner

from miaosuan.cli import app
from miaosuan.core.vocab import VOCAB_VERSION
from miaosuan.errors import BrokerUnavailableError
from miaosuan.ir.schema import Evidence, FactorPayload, Semantics, StrategySpec
from miaosuan.runtime.breaker import BreakerConfig, CircuitBreaker
from miaosuan.runtime.brokers.mt5 import MT5Broker
from miaosuan.runtime.live import LiveConfig, LiveRunner, _MagicScopedBroker, magic_from_spec
from miaosuan.runtime.scheduler import WatchItem
from miaosuan.runtime.signal_engine import SignalEngine
from miaosuan.runtime.state import kill_file_exists
from miaosuan.search.rpn import random_feasible

# ── 假 MetaTrader5 模块 ─────────────────────────────────────────────────────


class _FakeOrder:
    def __init__(self, retcode: int, order: int, comment: str = "ok") -> None:
        self.retcode = retcode
        self.order = order
        self.comment = comment


class _FakePosition:
    def __init__(
        self, ticket: int, symbol: str, is_long: bool, volume: float, price_open: float, magic: int
    ) -> None:
        self.ticket = ticket
        self.symbol = symbol
        self.type = 0 if is_long else 1  # POSITION_TYPE_BUY=0 / SELL=1
        self.volume = volume
        self.price_open = price_open
        self.magic = magic


class _FakeDeal:
    def __init__(
        self,
        ticket: int,
        order: int,
        symbol: str,
        is_buy: bool,
        volume: float,
        price: float,
        commission: float,
        swap: float,
        profit: float,
        ts: int,
        magic: int,
    ) -> None:
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
    def __init__(
        self,
        *,
        volume_step: float = 0.01,
        volume_min: float = 0.01,
        contract_size: float = 100.0,
        tick_value: float = 1.0,
        filling_mode: int = 1,
    ) -> None:
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

    def symbol_info_tick(self, symbol: str) -> Any:
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
        # 成交回执同步生成（真实 MT5 里经 history_deals_get 查询）
        self._next_deal += 1
        is_buy = int(req["type"]) == self.ORDER_TYPE_BUY
        fill_price = float(req["price"])
        self.deals.append(
            _FakeDeal(
                self._next_deal,
                self._next_order,
                req["symbol"],
                is_buy,
                float(req["volume"]),
                fill_price,
                -2.0,
                0.0,
                0.0,
                int(time.time()),
                int(req["magic"]),
            )
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


@pytest.fixture
def state_root(tmp_path: Path) -> Path:
    return tmp_path


def make_spec() -> StrategySpec:
    rng = np.random.default_rng(42)
    tokens = tuple(int(t) for t in random_feasible(rng, 8))
    return StrategySpec(
        name="live_smoke",
        payload=FactorPayload(tokens=tokens, vocab_version=VOCAB_VERSION),
        semantics=Semantics(timeframe="M30"),
        evidence=Evidence(gate_verdict="DEPLOYABLE"),
    )


# ── MT5Broker ───────────────────────────────────────────────────────────────


def test_broker_requires_connect_before_use() -> None:
    broker = MT5Broker()
    with pytest.raises(BrokerUnavailableError, match="未连接"):
        broker.positions()


def test_broker_submit_success_and_price_recorded(fake_mt5: _FakeMT5Module) -> None:
    broker = MT5Broker(login=1, password="x", server="s")
    broker.connect()
    ticket = broker.submit("XAUUSD", "buy", 0.1, magic=7, comment="signal")

    assert ticket.isdigit()
    assert len(fake_mt5.orders) == 1
    req = fake_mt5.orders[0]
    assert req["type"] == fake_mt5.ORDER_TYPE_BUY
    assert req["volume"] == pytest.approx(0.1)
    assert req["magic"] == 7
    assert req["deviation"] == MT5Broker.DEVIATION_POINTS
    assert req["price"] == pytest.approx(4000.2), "buy 用 ask 价"
    assert broker.last_request_price["XAUUSD"] == pytest.approx(4000.2)
    broker.close()


def test_broker_submit_rejection_raises(fake_mt5: _FakeMT5Module) -> None:
    broker = MT5Broker()
    broker.connect()
    fake_mt5.reject_next = 1
    with pytest.raises(BrokerUnavailableError, match="被拒"):
        broker.submit("XAUUSD", "buy", 0.1, magic=7)
    broker.close()


def test_broker_close_position_sends_hedging_position_field(
    fake_mt5: _FakeMT5Module,
) -> None:
    broker = MT5Broker()
    broker.connect()
    broker.submit("XAUUSD", "buy", 0.2, magic=7)
    fake_mt5.orders.clear()
    fake_mt5.positions.append(_FakePosition(888, "XAUUSD", True, 0.2, 4001.0, 7))

    broker.close_position("XAUUSD:888", reason="signal-flip")

    assert len(fake_mt5.orders) == 1
    req = fake_mt5.orders[0]
    assert req["position"] == 888, "hedging 平仓必须带 position ticket"
    assert req["type"] == fake_mt5.ORDER_TYPE_SELL, "多头平仓用反向卖单"
    assert req["volume"] == pytest.approx(0.2)
    assert req["price"] == pytest.approx(4000.0), "平多仓用 bid"
    broker.close()


def test_broker_positions_and_fills_mapping(fake_mt5: _FakeMT5Module) -> None:
    broker = MT5Broker()
    broker.connect()
    fake_mt5.positions.append(_FakePosition(777, "XAUUSD", False, 0.3, 4010.0, 7))
    fake_mt5.deals.append(
        _FakeDeal(
            900001, 500001, "XAUUSD", False, 0.3, 4009.0, -2.5, -1.0, 25.0, int(time.time()), 7
        )
    )

    positions = broker.positions()
    assert positions[0]["position_id"] == "777"
    assert positions[0]["side"] == "short"

    fills = broker.fills(0)
    assert fills[0]["side"] == "sell"
    assert fills[0]["fee"] == pytest.approx(-3.5), "fee = commission + swap"
    assert fills[0]["profit"] == pytest.approx(25.0)
    assert fills[0]["magic"] == 7
    broker.close()


def test_broker_symbol_spec(fake_mt5: _FakeMT5Module) -> None:
    broker = MT5Broker()
    broker.connect()
    spec = broker.symbol_spec("XAUUSD")
    assert spec.lot_step == pytest.approx(0.01)
    assert spec.min_lot == pytest.approx(0.01)
    assert spec.contract_size == pytest.approx(100.0)
    broker.close()


def test_broker_flatten_all_closes_everything(fake_mt5: _FakeMT5Module) -> None:
    broker = MT5Broker()
    broker.connect()
    fake_mt5.positions.append(_FakePosition(1, "XAUUSD", True, 0.1, 4000.0, 7))
    fake_mt5.positions.append(_FakePosition(2, "EURUSD", True, 0.1, 1.1, 7))
    broker.flatten_all(reason="kill")
    assert len(fake_mt5.orders) == 2, "账户级全平逐仓发出平仓单"
    broker.close()


# ── magic 域隔离（R4.2）─────────────────────────────────────────────────────


def test_magic_scoped_view_filters_positions_and_flatten(fake_mt5: _FakeMT5Module) -> None:
    broker = MT5Broker()
    broker.connect()
    fake_mt5.positions.append(_FakePosition(1, "XAUUSD", True, 0.1, 4000.0, 7))  # 我们的
    fake_mt5.positions.append(_FakePosition(2, "XAUUSD", True, 0.5, 4000.0, 42))  # 人工的
    scoped = _MagicScopedBroker(broker, magic=7)

    assert len(scoped.positions()) == 1, "只看见本域持仓"
    scoped.flatten_all(reason="kill")
    assert len(fake_mt5.orders) == 1, "只平本域仓位"
    assert fake_mt5.orders[0]["position"] == 1
    broker.close()


# ── 熔断（R4.3）─────────────────────────────────────────────────────────────


def test_breaker_trips_on_consecutive_rejections() -> None:
    breaker = CircuitBreaker(BreakerConfig(max_consecutive_rejections=3))
    breaker.on_order_rejected("r1")
    breaker.on_order_rejected("r2")
    assert not breaker.tripped
    breaker.on_order_accepted()  # 成功清零
    breaker.on_order_rejected("r3")
    breaker.on_order_rejected("r4")
    assert not breaker.tripped
    breaker.on_order_rejected("r5")
    assert breaker.tripped and "连续拒单" in breaker.reason


def test_breaker_trips_on_daily_loss() -> None:
    breaker = CircuitBreaker(BreakerConfig(max_daily_loss_usd=100.0))
    now = int(time.time())
    fills = [
        {"profit": -60.0, "fee": -3.0, "ts": now},
        {"profit": -50.0, "fee": -3.0, "ts": now},
    ]
    breaker.check_fills(fills)
    assert breaker.tripped and "单日已实现亏损" in breaker.reason


def test_breaker_trips_on_extreme_slippage() -> None:
    breaker = CircuitBreaker(BreakerConfig(max_slippage_ratio=0.005))
    now = int(time.time())
    fills = [{"price": 4100.0, "profit": 0.0, "fee": 0.0, "ts": now, "order_id": "500001"}]
    breaker.check_fills(fills, requested_prices={500001: 4000.0})  # 2.5% 滑点
    assert breaker.tripped and "极端滑点" in breaker.reason


def test_breaker_ignores_yesterday_loss() -> None:
    breaker = CircuitBreaker(BreakerConfig(max_daily_loss_usd=100.0))
    yesterday = int(time.time()) - 86400
    breaker.check_fills([{"profit": -500.0, "fee": -5.0, "ts": yesterday}])
    assert not breaker.tripped, "日亏按 UTC 日界聚合，昨日亏损不计入今日"


# ── LiveRunner 熔断接线 ─────────────────────────────────────────────────────


def _make_runner(fake_mt5_mod: _FakeMT5Module, state_root: Path) -> LiveRunner:

    spec = make_spec()
    engine = SignalEngine.__new__(SignalEngine)  # 不喂历史，仅携带 spec
    engine._spec = spec  # noqa: SLF001 - 测试装配
    engine._last_position = 0.0  # noqa: SLF001
    item = WatchItem(symbol="XAUUSD", timeframe="M30", engine=engine)
    broker = MT5Broker()
    broker.connect()
    return LiveRunner(
        item,
        port=None,  # type: ignore[arg-type] - 测试不驱动调度器
        profile_name="FOREX_XAUUSD_MT5",
        config=LiveConfig(
            risk_usd_per_bar=200.0,
            breaker=BreakerConfig(max_consecutive_rejections=2),
        ),
        broker=broker,
        state_root=state_root,
    )


def test_live_runner_trip_flow_creates_stop_and_flattens(
    fake_mt5: _FakeMT5Module, state_root: Path
) -> None:
    """连续拒单达上限 → 全平（本域）+ 创建 STOP + BrokerKillswitchError。"""
    runner = _make_runner(fake_mt5, state_root)
    fake_mt5.positions.append(_FakePosition(1, "XAUUSD", True, 0.1, 4000.0, runner.magic))

    from miaosuan.errors import BrokerKillswitchError

    runner.breaker.on_order_rejected("r1")
    runner.breaker.on_order_rejected("r2")
    assert runner.breaker.tripped
    with pytest.raises(BrokerKillswitchError, match="熔断"):
        runner._trip_flow()

    assert kill_file_exists(state_root), "熔断必须创建 STOP 文件持久化急停"
    close_orders = [o for o in fake_mt5.orders if o.get("position") == 1]
    assert close_orders, "熔断必须对本域持仓发出平仓单"
    assert all(p.magic == runner.magic for p in fake_mt5.positions), "只动本域，别的仓原样"


def test_magic_from_spec_stable() -> None:
    spec = make_spec()
    assert magic_from_spec(spec.spec_id) == magic_from_spec(spec.spec_id)
    assert 1 <= magic_from_spec(spec.spec_id) < 2**31


# ── CLI 真钱门禁（R4.4）─────────────────────────────────────────────────────


def test_cli_live_start_requires_confirmation(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["live", "start"])
    assert result.exit_code != 0
    assert "i-understand-real-money" in result.output


def test_cli_live_start_still_gated_with_watchlist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """即使 watchlist 就绪，缺确认标志仍必须拒绝。"""
    import miaosuan.runtime.state as state_mod
    import miaosuan.runtime.watchlist as watchlist_mod

    monkeypatch.setattr(state_mod, "artifacts_dir", lambda: str(tmp_path))
    monkeypatch.setattr(watchlist_mod, "artifacts_dir", lambda: str(tmp_path))
    runner = CliRunner()
    result = runner.invoke(app, ["live", "start"])
    assert result.exit_code != 0
