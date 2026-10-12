# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for full license text.

"""运行时端口：数据 / 执行 / 通知 三个可插拔抽象（路线图 R2.1 / Sample B）。

依赖方向约束（路线图 §3 铁律白名单）：本模块只 import 标准库；
具体实现（mt5/paper/feishu/shenji…）位于 ``runtime/sources|brokers|notifiers``，
运行时通过工厂/构造注入，**不感知具体实现**。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # 仅类型提示：端口层保持零运行时依赖（不 import market 层）
    from ..market.profiles import BrokerSpec

__all__ = ["Bar", "DataPort", "ExecutionPort", "NotifierPort"]


@dataclass(frozen=True)
class Bar:
    """一根已规范化行情 bar（契约同 ``data/panel.py``）。

    :param ts: UTC epoch 秒，bar **开盘**时间（panel.time 同口径）。
    :param closed: 必须为 ``True`` 才允许进入信号引擎——收盘纪律在端口层
        强制（runtime 只吃已收盘 bar，杜绝未收盘口径漂移）。
    """

    symbol: str
    timeframe: str
    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    closed: bool = True


class DataPort(ABC):
    """行情数据端口：拉取式推送**已收盘** Bar。"""

    @abstractmethod
    def stream_closed_bars(self, symbol: str, timeframe: str) -> Iterator[Bar]:
        """产出**自上次调用以来**新收盘的 Bar（升序、不重不漏后结束本轮）。

        拉取式契约：每次调用把自上次调用以来新收盘的 bar 逐根 yield 后
        结束，再次调用续拉（实现内部维护水位；首次调用回填可用历史）。
        bar 的 ``closed`` 必须可靠——未收盘 bar 不得产出（收盘纪律在
        端口层强制，调度层再按「周期已走完」双重校验）。
        """

    @abstractmethod
    def symbol_spec(self, symbol: str) -> BrokerSpec:
        """返回经纪商品种规格（tick_value / lot_step / min_lot / contract_size）。

        ``BrokerSpec`` 定义见 ``market/profiles.py``（画像可选字段 broker_spec）；
        R3/R4 执行侧消费，数据侧实现可暂抛 ``NotImplementedError``。
        """


class ExecutionPort(ABC):
    """执行端口：下单 / 平仓 / 持仓 / 成交回执（R3 Paper / R4 MT5 实现）。"""

    @abstractmethod
    def submit(self, symbol: str, side: str, lots: float, *, magic: int, comment: str = "") -> str:
        """开仓，返回 broker_order_id。实现必须快速返回或内部排队，不得阻塞调度循环。"""

    @abstractmethod
    def close_position(self, position_id: str, *, reason: str) -> None:
        """按持仓 id 平仓。

        MT5 hedging 账户（散户默认）下反向 ``submit`` 是**新开仓**而非平仓；
        平仓必须携带持仓 ticket（请求体 ``position`` 字段）。禁止用 submit 平仓。
        """

    @abstractmethod
    def positions(self) -> list[dict[str, Any]]:
        """当前持仓快照（方向/手数/开仓价/magic/ticket），供对账。"""

    @abstractmethod
    def fills(self, since_ts: int) -> list[dict[str, Any]]:
        """``since_ts`` 之后的成交回执，对账器（R3.5）的输入。

        Paper 实现读本地账本；MT5 实现读 ``history_deals_get``。
        没有回执通道就无法逐笔对账。
        """

    @abstractmethod
    def flatten_all(self, *, reason: str) -> None:
        """紧急全平（kill 开关/熔断路径），实现必须尽力而为并如实回报。"""

    def on_bar(self, bar: Bar) -> list[dict[str, Any]]:
        """每根收盘 bar 的撮合钩子（可选实现）。

        PaperBroker 用它完成"下一根开盘"的挂起撮合并返回本次成交回执；
        真实券商实现（MT5）成交即时回报、无本地撮合，默认空实现。
        调度回路对每个 broker 统一调用，多态分派。
        """
        return []


class NotifierPort(ABC):
    """通知端口：转折/告警推送（log 兜底，R5 飞书）。"""

    @abstractmethod
    def notify(self, title: str, body: str, *, level: str = "info") -> None: ...
