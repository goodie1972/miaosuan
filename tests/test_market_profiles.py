# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""``MarketProfile`` / ``CostModel`` 单测（M8）。

覆盖：v1 全 14 必填字段冻结、四实例齐备、``apply_cost``（含买卖不对称）、``CostModel``
边界、``long_only`` 开关（A 股语义）无负仓位。
"""

from __future__ import annotations

import numpy as np
import pytest

from miaosuan.core.signal import compute_target_positions
from miaosuan.errors import ConfigError
from miaosuan.market.cost import ZERO_COST, CostModel
from miaosuan.market.profiles import (
    CN_COMMODITY_FUTURES,
    CN_EQUITY_RESEARCH,
    CRYPTO_BTC,
    EXPECTED_PROFILE_NAMES,
    FOREX_XAUUSD,
    MARKET_PROFILE_FIELDS,
    MARKET_PROFILE_REQUIRED_FIELDS,
    PROFILES,
    US_EQUITY_RESEARCH,
    FrozenMarketProfile,
    get_profile,
)

# ── v1 字段冻结 ─────────────────────────────────────────────────────────────


def test_all_profiles_present() -> None:
    assert tuple(PROFILES) == EXPECTED_PROFILE_NAMES
    assert len(PROFILES) == 5


def test_v1_field_set_frozen() -> None:
    assert len(MARKET_PROFILE_REQUIRED_FIELDS) == 14, "v1 必须为 14 必填字段"
    assert len(MARKET_PROFILE_FIELDS) == 15
    assert MARKET_PROFILE_FIELDS[-1] == "limit_pct"


@pytest.mark.parametrize("name", EXPECTED_PROFILE_NAMES)
def test_every_profile_has_all_v1_fields(name: str) -> None:
    profile = get_profile(name)
    for field in MARKET_PROFILE_REQUIRED_FIELDS:
        assert hasattr(profile, field), f"{name} 缺字段 {field}"
    assert hasattr(profile, "limit_pct")
    assert isinstance(profile, FrozenMarketProfile)


def test_forex_xauusd_values() -> None:
    assert FOREX_XAUUSD.name == "FOREX_XAUUSD"
    assert FOREX_XAUUSD.quote_currency == "USD"
    assert FOREX_XAUUSD.long_only is False
    assert FOREX_XAUUSD.leverage > 1.0
    assert FOREX_XAUUSD.bars_per_year == 6240
    assert FOREX_XAUUSD.bars_per_day == 24
    assert FOREX_XAUUSD.allow_intraday_exit is True
    assert FOREX_XAUUSD.settlement_days == 0
    assert FOREX_XAUUSD.limit_pct is None


def test_cn_equity_research_semantics() -> None:
    assert CN_EQUITY_RESEARCH.long_only is True
    assert CN_EQUITY_RESEARCH.allow_intraday_exit is False, "A 股 T+1"
    assert CN_EQUITY_RESEARCH.settlement_days == 1
    assert CN_EQUITY_RESEARCH.limit_pct == 0.10
    assert CN_EQUITY_RESEARCH.quote_currency == "CNY"


def test_crypto_btc_is_fourth_placeholder() -> None:
    assert EXPECTED_PROFILE_NAMES[3] == "CRYPTO_BTC"
    assert CRYPTO_BTC.quote_currency == "USDT"
    assert CRYPTO_BTC.bars_per_day == 24
    assert CRYPTO_BTC.settlement_days == 0


def test_us_equity_research_placeholder() -> None:
    assert US_EQUITY_RESEARCH.name == "US_EQUITY_RESEARCH"
    assert US_EQUITY_RESEARCH.settlement_days == 1


def test_cn_commodity_futures_semantics() -> None:
    """国内商品期货：T+0 可当日平仓、保证金杠杆、数据源严格匹配 AkShare。"""
    assert CN_COMMODITY_FUTURES.name == "CN_COMMODITY_FUTURES"
    assert CN_COMMODITY_FUTURES.quote_currency == "CNY"
    assert CN_COMMODITY_FUTURES.allow_intraday_exit is True, "期货 T+0"
    assert CN_COMMODITY_FUTURES.leverage > 1.0, "保证金杠杆"
    assert CN_COMMODITY_FUTURES.long_only is False
    assert CN_COMMODITY_FUTURES.settlement_days == 0
    assert CN_COMMODITY_FUTURES.data_sources == ["AkShare"]


def test_shenji_only_for_forex_xauusd() -> None:
    """妙算库仅存 XAUUSD 单一品种，故只归属 FOREX_XAUUSD，不得出现在其他画像。"""
    assert "Shenji" in FOREX_XAUUSD.data_sources
    for profile in (
        CN_EQUITY_RESEARCH,
        US_EQUITY_RESEARCH,
        CRYPTO_BTC,
        CN_COMMODITY_FUTURES,
    ):
        assert "Shenji" not in profile.data_sources, f"{profile.name} 不应含妙算源"


def test_get_profile_unknown_raises() -> None:
    with pytest.raises(ConfigError):
        get_profile("NOPE")


# ── CostModel ───────────────────────────────────────────────────────────────


def test_cost_model_symmetric() -> None:
    cm = CostModel(commission=0.0001, slippage=0.0002)
    assert cm.buy_rate() == pytest.approx(0.0003)
    assert cm.sell_rate() == pytest.approx(0.0003)
    assert cm.total("buy") == pytest.approx(0.0003)
    assert cm.total("sell") == pytest.approx(0.0003)


def test_cost_model_asymmetric_sell_tax() -> None:
    cm = CostModel(commission=0.0001, slippage=0.0002, sell_tax=0.001, asymmetric=True)
    assert cm.buy_rate() == pytest.approx(0.0003)
    assert cm.sell_rate() == pytest.approx(0.0013)
    assert cm.round_trip_rate() == pytest.approx(0.0016)


def test_cost_model_rejects_bad_side_and_negative() -> None:
    with pytest.raises(ConfigError):
        CostModel(commission=-0.1)
    with pytest.raises(ConfigError):
        CostModel().total("hold")


def test_cost_model_scaled() -> None:
    cm = CostModel(commission=0.0001, slippage=0.0002, sell_tax=0.001, asymmetric=True)
    doubled = cm.scaled(2.0)
    assert doubled.commission == pytest.approx(0.0002)
    assert doubled.sell_rate() == pytest.approx(cm.sell_rate() * 2.0)


# ── apply_cost ──────────────────────────────────────────────────────────────


def _profile(cost_model: object) -> FrozenMarketProfile:
    return FrozenMarketProfile(
        name="TEST",
        quote_currency="USD",
        cost_model=cost_model,  # type: ignore[arg-type]
        long_only=False,
        leverage=1.0,
        lot_size=1.0,
        tick_size=0.01,
        contract_multiplier=1.0,
        bars_per_year=6240,
        bars_per_day=24,
        session_hours=((0, 24),),
        tradable_mask=None,
        allow_intraday_exit=True,
        settlement_days=0,
        limit_pct=None,
    )


def test_apply_cost_zero_cost_is_gross() -> None:
    profile = _profile(ZERO_COST)
    position = np.array([[0.0, 1.0, -1.0, 0.5]], dtype=np.float64)
    ret = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float64)
    net = profile.apply_cost(position, ret)
    assert np.allclose(net, position * ret)


def test_apply_cost_asymmetric_charges_only_on_decrease() -> None:
    cm = CostModel(commission=0.0, slippage=0.0, sell_tax=0.001, asymmetric=True)
    profile = _profile(cm)
    position = np.array([[0.0, 1.0, 1.0, 0.0]], dtype=np.float64)
    ret = np.zeros((1, 4), dtype=np.float64)
    net = profile.apply_cost(position, ret)
    # 建仓（increase）成本率 0；平仓（decrease）成本率 0.001
    assert net[0, 1] == pytest.approx(0.0)
    assert net[0, 3] == pytest.approx(-0.001)


def test_apply_cost_shape_mismatch() -> None:
    profile = _profile(ZERO_COST)
    with pytest.raises(ConfigError):
        profile.apply_cost(np.zeros((1, 4)), np.zeros((1, 5)))


def test_profile_validation_rejects_bad_values() -> None:
    with pytest.raises(ConfigError):
        FrozenMarketProfile(
            name="X", quote_currency="USD", cost_model=ZERO_COST, long_only=False,
            leverage=0.0, lot_size=1.0, tick_size=0.01, contract_multiplier=1.0,
            bars_per_year=100, bars_per_day=1, session_hours=((0, 24),),
            tradable_mask=None, allow_intraday_exit=True, settlement_days=0,
        )


# ── long_only 开关（A 股语义）──────────────────────────────────────────────


def test_long_only_switch_has_no_negative_positions() -> None:
    rng = np.random.default_rng(2026)
    factors = rng.normal(0.0, 1.0, (2, 200)).astype(np.float32)
    short_enabled = compute_target_positions(factors, long_only=False)
    assert np.any(short_enabled < 0.0), "long_only=False 应允许负仓位"

    long_only = compute_target_positions(factors, long_only=True)
    assert np.all(long_only >= 0.0), "long_only=True 不应出现负仓位"
    assert np.any(long_only > 0.0)


def test_cn_profile_long_only_drives_signal() -> None:
    """A 股 profile 的 long_only 直接驱动 signal 开关（产品轴 → 映射）。"""
    rng = np.random.default_rng(11)
    factors = rng.normal(0.0, 1.0, (1, 300)).astype(np.float32)
    pos = compute_target_positions(
        factors, min_trade_exposure=0.05, long_only=CN_EQUITY_RESEARCH.long_only
    )
    assert np.all(pos >= 0.0)
