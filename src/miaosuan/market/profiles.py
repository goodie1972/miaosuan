# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""``MarketProfile`` 实例（架构 §2 ``market/profiles.py``，M8）。

**产品轴「市场怎么交易」的落地**：成本 / 多空 / 杠杆 / 合约规格 / 时段 / 可交易性 / 结算。
这些**都不属于数据层**；复权与日历归 :mod:`miaosuan.data.sources`（架构 §9.4 硬分界）。

本模块提供 :class:`FrozenMarketProfile`（v1 全 14 必填字段 + 1 可选 ``limit_pct``，冻结）与
四个实例：

======================  ==================  ==========================================
实例                     市场                 说明
======================  ==================  ==========================================
``FOREX_XAUUSD``         外汇/贵金属现货      MVP 目标市场（H1，24×5）
``CN_EQUITY_RESEARCH``   A 股研究            占位：``long_only=True``，T+1，涨跌停 ±10%
``US_EQUITY_RESEARCH``   美股研究            占位：T+1（``settlement_days=1``）
``CRYPTO_BTC``          加密现货             ★第四个占位（24/7，USDT 计价）
======================  ==================  ==========================================

**零改主干证据**：新增以上任一 profile **不触碰 ``core/``**（CI 断言见
``tests/test_core_zero_touch.py``）。实现结构性地满足 ``core/ports.py`` 的
:class:`~miaosuan.core.ports.MarketProfile` 协议。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..errors import ConfigError
from .cost import CostModel

__all__ = [
    "CRYPTO_BTC",
    "CN_EQUITY_RESEARCH",
    "EXPECTED_PROFILE_NAMES",
    "FOREX_XAUUSD",
    "FrozenMarketProfile",
    "MARKET_PROFILE_FIELDS",
    "PROFILES",
    "US_EQUITY_RESEARCH",
    "get_profile",
]

#: ``MarketProfile`` v1 的字段名（14 必填 + 1 可选，冻结；架构 §1.2.1.5）
MARKET_PROFILE_FIELDS: tuple[str, ...] = (
    # 交易规则
    "name",
    "quote_currency",
    "cost_model",
    "long_only",
    "leverage",
    # 合约规格
    "lot_size",
    "tick_size",
    "contract_multiplier",
    # 时钟与时段
    "bars_per_year",
    "bars_per_day",
    "session_hours",
    # 可交易性与结算
    "tradable_mask",
    "allow_intraday_exit",
    "settlement_days",
    # 可选
    "limit_pct",
)

#: v1 的必填字段（14 个；``limit_pct`` 为可选）
MARKET_PROFILE_REQUIRED_FIELDS: tuple[str, ...] = MARKET_PROFILE_FIELDS[:-1]


@dataclass(frozen=True)
class FrozenMarketProfile:
    """冻结的 :class:`~miaosuan.core.ports.MarketProfile` 实例（v1，14 必填 + 1 可选）。

    :param name: 画像名（``FOREX_XAUUSD`` / ...）。
    :param quote_currency: 计价币种（``USD`` / ``CNY`` / ``USDT``）。
    :param cost_model: 成本模型（买卖可不对称）。类型取具体 :class:`CostModel`（frozen 值对象），
        它是 ``core/ports.py`` 中 ``CostModelProtocol`` 在 v1 的唯一实现；``MarketProfile`` 协议
        对成本模型的约定在运行时由结构（``total`` 方法）满足。
    :param long_only: 是否只做多（A 股 = True）。
    :param leverage: 杠杆（``1.0`` = 无杠杆）。
    :param lot_size: 最小手数。
    :param tick_size: 最小变动价位。
    :param contract_multiplier: 合约乘数。
    :param bars_per_year: 每年 bar 数（年化换算）。
    :param bars_per_day: 每日 bar 数（warm-up 天数口径）。
    :param session_hours: 交易时段 ``((start_hour, start_min), (end_hour, end_min), ...)``；
        ``((0, 24),)`` 表示 24 小时连续。
    :param tradable_mask: 逐 bar 可交易掩码 ``[T]`` bool（``None`` 表示全部可交易）。
    :param allow_intraday_exit: 是否允许当日平仓（A 股 T+1 = False）。
    :param settlement_days: 结算延迟天数（A 股 = 1；外汇/加密 = 0）。
    :param limit_pct: 涨跌停幅度（A 股 0.10 / 0.20 / 0.05）；无涨跌停市场 = None。
    :param symbols: Web UI 便捷下拉项——该画像覆盖的品种标识（如 ``["XAUUSD"]``）。
        非 v1 交易规则契约，仅供数据获取页的品种下拉；留空则 UI 不预填选项。
    :param timeframes: Web UI 便捷下拉项——该画像支持的周期标识（如 ``["H1", "D1"]``）。
        同上，仅供数据获取页的周期下拉。
    :param data_sources: Web UI 数据获取页——该市场画像**严格匹配**的数据源类型列表
        （取值为 ``Shenji`` / ``TradingView`` / ``OKX`` / ``Binance`` / ``Dukascopy``
        / ``AkShare`` / ``其他``）。
        不同市场的数据源相互独立、互不相同；UI 仅展示该列表内的来源，杜绝跨市场串用。
        注：妙算（``Shenji``）库仅存 XAUUSD 单一品种，故只归属 FOREX_XAUUSD。
    """

    name: str
    quote_currency: str
    cost_model: CostModel
    long_only: bool
    leverage: float
    lot_size: float
    tick_size: float
    contract_multiplier: float
    bars_per_year: int
    bars_per_day: int
    session_hours: tuple[tuple[int, int], ...]
    tradable_mask: np.ndarray | None
    allow_intraday_exit: bool
    settlement_days: int
    limit_pct: float | None = None
    # ── UI 选择便捷字段（非 v1 交易规则契约，仅供 Web UI 下拉联动）────
    symbols: list[str] = field(default_factory=list)
    timeframes: list[str] = field(default_factory=list)
    # ── 数据源归类（非 v1 交易规则契约，仅供数据获取页按市场画像严格匹配）──
    data_sources: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.name:
            raise ConfigError("MarketProfile.name 不得为空")
        if not self.quote_currency:
            raise ConfigError(f"[{self.name}] quote_currency 不得为空")
        if self.leverage <= 0:
            raise ConfigError(f"[{self.name}] leverage 必须为正", context={"leverage": self.leverage})
        if self.lot_size <= 0:
            raise ConfigError(f"[{self.name}] lot_size 必须为正")
        if self.tick_size <= 0:
            raise ConfigError(f"[{self.name}] tick_size 必须为正")
        if self.contract_multiplier <= 0:
            raise ConfigError(f"[{self.name}] contract_multiplier 必须为正")
        if self.bars_per_year <= 0 or self.bars_per_day <= 0:
            raise ConfigError(f"[{self.name}] bars_per_year / bars_per_day 必须为正")
        if self.settlement_days < 0:
            raise ConfigError(f"[{self.name}] settlement_days 不得为负")
        if not self.session_hours:
            raise ConfigError(f"[{self.name}] session_hours 不得为空")
        if self.limit_pct is not None and not 0.0 < self.limit_pct < 1.0:
            raise ConfigError(f"[{self.name}] limit_pct 必须在 (0, 1) 内")

    # ── 协议方法 ────────────────────────────────────────────────────────────

    def apply_cost(self, position: np.ndarray, ret: np.ndarray) -> np.ndarray:
        """按成本模型把仓位毛收益换算为净收益（买卖方向各自计费，支持不对称）。

        ``net = position * ret - (buy_rate * Δpos⁺ + sell_rate * Δpos⁻)``，
        其中 ``Δpos`` 为相邻 bar 的仓位变化（首 bar 相对 0 仓位）。

        :param position: ``[N, T]`` 目标仓位（分数）。
        :param ret: ``[N, T]`` 标的收益。
        """
        pos = np.asarray(position, dtype=np.float64)
        r = np.asarray(ret, dtype=np.float64)
        if pos.shape != r.shape:
            raise ConfigError(
                "apply_cost: position 与 ret 形状必须一致",
                context={"position": pos.shape, "ret": r.shape},
            )
        gross = pos * r
        prev = np.zeros_like(pos)
        if pos.shape[-1] > 1:
            prev[:, 1:] = pos[:, :-1]
        delta = pos - prev
        increase = np.asarray(np.clip(delta, 0.0, None), dtype=np.float64)
        decrease = np.asarray(np.clip(-delta, 0.0, None), dtype=np.float64)
        cost = self.cost_model.total("buy") * increase + self.cost_model.total("sell") * decrease
        return np.asarray(gross - cost, dtype=np.float64)

    def is_tradable(self, index: int) -> bool:
        """第 ``index`` 根 bar 是否可交易（无掩码则视为可交易）。"""
        if self.tradable_mask is None:
            return True
        return bool(self.tradable_mask[index])


# ── 五个实例（FOREX_XAUUSD + 3 个占位 + 国内商品期货）─────────────────────

#: 外汇/贵金属现货（MVP 目标市场）。H1，24×5 交易，年化 bar 数对齐 AM ``6240``。
FOREX_XAUUSD = FrozenMarketProfile(
    name="FOREX_XAUUSD",
    quote_currency="USD",
    cost_model=CostModel(commission=0.0001, slippage=0.0002, sell_tax=0.0, asymmetric=False),
    long_only=False,
    leverage=100.0,
    lot_size=0.01,
    tick_size=0.01,
    contract_multiplier=100.0,
    bars_per_year=6240,
    bars_per_day=24,
    session_hours=((0, 0), (0, 24)),
    tradable_mask=None,
    allow_intraday_exit=True,
    settlement_days=0,
    limit_pct=None,
    symbols=["XAUUSD"],
    timeframes=["D1", "H1", "H4", "M15", "M30", "M5", "W1"],
    data_sources=["Shenji", "TradingView", "Dukascopy", "MT4"],
)

#: A 股研究（**占位**）：只做多、T+1、涨跌停 ±10%、卖出印花税。
CN_EQUITY_RESEARCH = FrozenMarketProfile(
    name="CN_EQUITY_RESEARCH",
    quote_currency="CNY",
    cost_model=CostModel(commission=0.00025, slippage=0.001, sell_tax=0.0005, asymmetric=True),
    long_only=True,
    leverage=1.0,
    lot_size=100.0,
    tick_size=0.01,
    contract_multiplier=1.0,
    bars_per_year=244 * 4,
    bars_per_day=4,
    session_hours=((9, 30), (11, 30), (13, 0), (15, 0)),
    tradable_mask=None,
    allow_intraday_exit=False,
    settlement_days=1,
    limit_pct=0.10,
    symbols=["000001"],
    timeframes=["D1", "W1", "M1", "M5", "M15", "M30", "H1"],
    data_sources=["AkShare"],
)

#: 美股研究（**占位**）：T+1 结算，6.5 小时交易时段。
US_EQUITY_RESEARCH = FrozenMarketProfile(
    name="US_EQUITY_RESEARCH",
    quote_currency="USD",
    cost_model=CostModel(commission=0.0001, slippage=0.0005, sell_tax=0.0, asymmetric=False),
    long_only=False,
    leverage=1.0,
    lot_size=1.0,
    tick_size=0.01,
    contract_multiplier=1.0,
    bars_per_year=252 * 7,
    bars_per_day=7,
    session_hours=((9, 30), (16, 0)),
    tradable_mask=None,
    allow_intraday_exit=True,
    settlement_days=1,
    limit_pct=None,
    symbols=["AAPL"],
    timeframes=["D1", "W1", "M1", "M5", "M15", "M30", "H1"],
    data_sources=["TradingView", "AkShare"],
)

#: 加密现货（**第四个占位**）：24/7 连续，USDT 计价，无涨跌停。
CRYPTO_BTC = FrozenMarketProfile(
    name="CRYPTO_BTC",
    quote_currency="USDT",
    cost_model=CostModel(commission=0.0004, slippage=0.0005, sell_tax=0.0, asymmetric=False),
    long_only=False,
    leverage=1.0,
    lot_size=1e-05,
    tick_size=0.1,
    contract_multiplier=1.0,
    bars_per_year=24 * 365,
    bars_per_day=24,
    session_hours=((0, 0), (0, 24)),
    tradable_mask=None,
    allow_intraday_exit=True,
    settlement_days=0,
    limit_pct=None,
    symbols=["BTCUSDT"],
    timeframes=["D1", "H1", "H4", "M15", "M30", "M5", "W1"],
    data_sources=["OKX", "TradingView", "Binance"],
)

#: 国内商品期货（化工 / 农化方向）。T+0 可当日平仓、保证金杠杆（约 10x）、
#: 涨跌停幅度随品种 / 交易所浮动故 ``limit_pct=None``。
#: 合约乘数取 10 吨/手（甲醇 MA 口径；尿素 UR 为 20 吨/手，此处按主品种近似）。
CN_COMMODITY_FUTURES = FrozenMarketProfile(
    name="CN_COMMODITY_FUTURES",
    quote_currency="CNY",
    cost_model=CostModel(commission=0.0001, slippage=0.0002, sell_tax=0.0, asymmetric=False),
    long_only=False,
    leverage=10.0,
    lot_size=1.0,
    tick_size=1.0,
    contract_multiplier=10.0,
    bars_per_year=244 * 4,
    bars_per_day=4,
    session_hours=((9, 0), (11, 30), (13, 30), (15, 0)),
    tradable_mask=None,
    allow_intraday_exit=True,
    settlement_days=0,
    limit_pct=None,
    symbols=["MA0", "UR0"],
    timeframes=["D1", "H1", "M15", "M30", "M5"],
    data_sources=["AkShare"],
)

#: 全部 profile 注册表（新增产品只需在此追加，**不改 ``core/``**）
PROFILES: dict[str, FrozenMarketProfile] = {
    FOREX_XAUUSD.name: FOREX_XAUUSD,
    CN_EQUITY_RESEARCH.name: CN_EQUITY_RESEARCH,
    US_EQUITY_RESEARCH.name: US_EQUITY_RESEARCH,
    CRYPTO_BTC.name: CRYPTO_BTC,
    CN_COMMODITY_FUTURES.name: CN_COMMODITY_FUTURES,
}

#: 期望存在的 profile 名（供 CI「零改主干」与完整性断言使用）
EXPECTED_PROFILE_NAMES: tuple[str, ...] = (
    "FOREX_XAUUSD",
    "CN_EQUITY_RESEARCH",
    "US_EQUITY_RESEARCH",
    "CRYPTO_BTC",
    "CN_COMMODITY_FUTURES",
)


def get_profile(name: str) -> FrozenMarketProfile:
    """按名取 profile；未知名抛 :class:`ConfigError`。"""
    try:
        return PROFILES[name]
    except KeyError as exc:
        raise ConfigError(
            f"未知 MarketProfile: {name!r}（可选 {sorted(PROFILES)}）", context={"name": name}
        ) from exc


def all_profiles() -> dict[str, FrozenMarketProfile]:
    """返回全部 profile 的浅拷贝（名称 → 实例）。"""
    return dict(PROFILES)
