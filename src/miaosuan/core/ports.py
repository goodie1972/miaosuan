"""核心域端口协议（依赖倒置的接口面，架构 §1.2.1.2 / §1.2）。

本模块只声明**协议**（``typing.Protocol``），不含任何实现、不做 IO、不读环境：

* :class:`MarketProfile` —— 产品轴「怎么交易」（14 字段 v1 + ``apply_cost``）；
* :class:`CostModelProtocol` —— 交易成本模型（买卖可不对称）；
* :class:`DataSource` —— 产品轴「数据怎么来」（复权 / 日历 / 可交易掩码）；
* :class:`TargetPort` —— 平台轴双向编译器（IR ⇄ 目标平台策略）。

作用：让 ``core/`` 的 ``evaluator`` / ``backtest`` / ``signal`` 通过**协议**消费数据与
市场参数（依赖倒置），从而对「模式轴 / 产品轴 / 平台轴」三者不感知。T02 会提供
``market/profiles.py`` 的 ``FOREX_XAUUSD`` 实例与 ``data/sources/parquet_mt.py`` 的具体
实现；M5 仅**预留接口面**，默认实现（直接参数注入）与 AM 逐点一致。

设计说明：

* 协议方法的函数体**只有 docstring**（Protocol 成员不实现逻辑），因此既不引入
  占位标记，也不违反「core 无 IO / 无 torch」铁律。
* 所有形状注解沿用统一契约：``[N, T]``，N=截面/样本，T=时间。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

__all__ = [
    "CostModelProtocol",
    "DataSource",
    "MarketProfile",
    "TargetPort",
]


@runtime_checkable
class CostModelProtocol(Protocol):
    """交易成本模型（架构类图 `CostModel`；买卖可不对称）。"""

    commission: float
    slippage: float
    sell_tax: float
    asymmetric: bool

    def total(self, side: str, notional: float) -> float:
        """返回某一方向的单边总成本率（佣金 + 滑点 [+ 卖出税]）。

        :param side: ``"buy"`` 或 ``"sell"``。
        :param notional: 名义金额。
        """


@runtime_checkable
class MarketProfile(Protocol):
    """产品轴协议：市场怎么交易（14 字段 v1，架构 §1.2.1.5）。"""

    name: str
    quote_currency: str
    cost_model: CostModelProtocol
    long_only: bool
    leverage: float
    lot_size: float
    tick_size: float
    contract_multiplier: float
    bars_per_year: int
    bars_per_day: int
    session_hours: tuple[int, ...]
    tradable_mask: np.ndarray | None
    allow_intraday_exit: bool
    settlement_days: int
    limit_pct: float | None

    def apply_cost(self, position: np.ndarray, ret: np.ndarray) -> np.ndarray:
        """把仓位下的毛收益按成本模型换算为净收益（含杠杆 / 合约乘数）。

        :param position: ``[N, T]`` 目标仓位。
        :param ret: ``[N, T]`` 标的收益。
        """


@runtime_checkable
class DataSource(Protocol):
    """产品轴协议：数据怎么来（复权 / 日历 / 时间索引归此层，架构 §1.2.1.2）。"""

    source_id: str
    symbols: list[str]
    timeframe: str

    def load(self, spec: dict[str, object]) -> object:
        """加载数据，返回 ``Panel``（``data/panel.py``，T02 实现）。

        :param spec: 数据规格（文件、品种、周期、复权模式等）。
        """

    def adjustment_mode(self) -> str:
        """返回复权模式（如 ``"none"`` / ``"forward"`` / ``"backward"``）。"""

    def trading_calendar(self) -> np.ndarray:
        """返回交易日历（Unix 秒时间戳数组）。"""

    def tradable_mask(self) -> np.ndarray:
        """返回可交易掩码（``[T]`` bool）。"""


@runtime_checkable
class TargetPort(Protocol):
    """平台轴协议：目标平台双向编译器（架构 §1.2.1.6）。"""

    platform_id: str
    language: str
    bar_semantics: str
    signal_contract: type
    naming_pattern: str
    stub_modules: dict[str, str]

    def compile_from_ir(self, spec: object) -> str:
        """正向：``FactorSpec`` / 策略 IR → 目标平台策略源码。"""

    def extract_to_ir(self, source: str) -> object:
        """反向：目标平台策略源码 → ``ParamSpace`` / IR。"""
