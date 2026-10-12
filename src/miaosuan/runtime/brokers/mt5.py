# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for full license text.

"""MT5Broker（路线图 R4.1）：真实执行通道（``[live]`` extra，仅 Windows）。

与 :class:`~miaosuan.runtime.brokers.paper.PaperBroker` 的本质区别：成交由
**券商**回报，本地不再模拟撮合——提交即市价 DEAL（带 price/deviation/
type_time），回执走 ``history_deals_get``，平仓必须携带持仓 ticket
（hedging 账户反向 submit 是新开仓而非平仓）。

安全约定：

* **凭证一律经构造参数注入**（CLI ``live start`` 是唯一 env 边界，读
  ``MIAOSUAN_MT5_LOGIN / MIAOSUAN_MT5_PASSWORD / MIAOSUAN_MT5_SERVER``，
  gitleaks 红线：任何路径都不得硬编码或落盘）；
* ``MetaTrader5`` 包**方法内惰性导入**，缺依赖（非 Windows / 未装
  ``miaosuan[live]``）时构造即抛 :class:`BrokerUnavailableError`；
* 终端会话由 :meth:`connect` / :meth:`close` 显式管理（运行器 start/stop
  调用），不跨实例持有；
* 订单请求失败（retcode != DONE）抛 :class:`BrokerUnavailableError`——
  连续拒单的熔断计数由运行器承担。
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from ...errors import BrokerUnavailableError
from ...market.profiles import BrokerSpec
from ..ports import ExecutionPort

__all__ = ["MT5Broker"]

logger = logging.getLogger(__name__)


class MT5Broker(ExecutionPort):
    """MetaTrader5 真实执行通道（DEAL 市价单；hedging 平仓带 position ticket）。"""

    #: 允许的滑点（点）；市价单保护性偏差
    DEVIATION_POINTS = 20

    def __init__(
        self,
        *,
        terminal_path: str | None = None,
        login: int | None = None,
        password: str | None = None,
        server: str | None = None,
        now_fn: Callable[[], float] = time.time,
    ) -> None:
        self._terminal_path = terminal_path
        self._login = login
        self._password = password
        self._server = server
        self._now_fn = now_fn
        self._mt5: Any | None = None
        self._connected = False
        #: 最近一次订单请求价（按品种；滑点核查对照）
        self.last_request_price: dict[str, float] = {}

    # ── 会话管理（运行器 start/stop 调用）────────────────────────────────

    def connect(self) -> None:
        """初始化终端会话（幂等；缺依赖/未运行终端如实报错）。"""
        if self._connected:
            return
        mt5 = self._load_mt5()
        kwargs: dict[str, Any] = {}
        if self._terminal_path:
            kwargs["path"] = self._terminal_path
        if self._login is not None:
            kwargs["login"] = int(self._login)
            if self._password:
                kwargs["password"] = self._password
            if self._server:
                kwargs["server"] = self._server
        if not mt5.initialize(**kwargs):
            raise BrokerUnavailableError(
                f"MT5 终端初始化失败: {mt5.last_error()}（终端未运行或未登录?）",
                context={"terminal_path": self._terminal_path or "(默认)"},
            )
        self._mt5 = mt5
        self._connected = True
        logger.info("MT5 执行通道已连接（login=%s）", self._login or "(终端已登录账户)")

    def close(self) -> None:
        """结束终端会话（尽力而为）。"""
        if self._mt5 is not None and self._connected:
            with contextlib.suppress(Exception):  # 关闭尽力而为
                self._mt5.shutdown()
        self._connected = False
        self._mt5 = None

    def _require_mt5(self) -> Any:
        if not self._connected or self._mt5 is None:
            raise BrokerUnavailableError("MT5 执行通道未连接（先调用 connect()）")
        return self._mt5

    @staticmethod
    def _load_mt5() -> Any:
        try:
            import MetaTrader5 as mt5  # 仅 [live] extra 提供
        except ImportError as exc:
            raise BrokerUnavailableError(
                "缺少 MetaTrader5 包: pip install 'miaosuan[live]'（仅 Windows）"
            ) from exc
        return mt5

    # ── ExecutionPort ─────────────────────────────────────────────────────

    def submit(self, symbol: str, side: str, lots: float, *, magic: int, comment: str = "") -> str:
        """市价开仓（DEAL），返回订单 ticket。拒单抛 :class:`BrokerUnavailableError`。

        参数校验与 PaperBroker 同口径（对抗性测试回归）：方向非法 / 手数
        非正 / 低于 volume_min / 不对齐 volume_step 一律本地拒绝，**绝不**
        把坏参数发给券商（券商拒绝要付网络往返 + 触发拒单熔断计数）。
        """
        mt5 = self._require_mt5()
        if side not in ("buy", "sell"):
            raise BrokerUnavailableError(f"非法方向: {side!r}", context={"side": side})
        if lots <= 0:
            raise BrokerUnavailableError(
                "手数必须为正", context={"lots": lots}
            )
        info = mt5.symbol_info(symbol)
        if info is None:
            raise BrokerUnavailableError(
                f"品种不存在（券商侧）: {symbol}", context={"symbol": symbol}
            )
        min_lot = float(getattr(info, "volume_min", 0.0) or 0.0)
        step = float(getattr(info, "volume_step", 0.0) or 0.0)
        if min_lot and lots + 1e-9 < min_lot:
            raise BrokerUnavailableError(
                f"手数低于 volume_min（不交易，禁止向上取整）",
                context={"lots": lots, "min_lot": min_lot},
            )
        if step:
            steps = round(lots / step)
            if abs(steps * step - lots) > 1e-9:
                raise BrokerUnavailableError(
                    "手数未按 volume_step 对齐",
                    context={"lots": lots, "volume_step": step},
                )
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise BrokerUnavailableError(
                f"无法获取 {symbol} 行情（品种不在 Market Watch?）",
                context={"symbol": symbol},
            )
        req: dict[str, Any] = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(lots),
            "type": mt5.ORDER_TYPE_BUY if side == "buy" else mt5.ORDER_TYPE_SELL,
            "price": float(tick.ask if side == "buy" else tick.bid),  # 市价单参考价
            "deviation": self.DEVIATION_POINTS,
            "magic": int(magic),
            "comment": comment[:31],  # MT5 注释长度上限
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._filling_mode(symbol),
        }
        result = mt5.order_send(req)
        if result is None:
            raise BrokerUnavailableError(
                f"order_send 返回 None: {mt5.last_error()}",
                context={"symbol": symbol, "side": side, "lots": lots},
            )
        retcode = int(getattr(result, "retcode", -1))
        if retcode != int(mt5.TRADE_RETCODE_DONE):
            raise BrokerUnavailableError(
                f"order_send 被拒: retcode={retcode} comment={getattr(result, 'comment', '')}",
                context={"symbol": symbol, "side": side, "lots": lots, "retcode": retcode},
            )
        self.last_request_price[symbol] = float(req["price"])
        ticket = str(getattr(result, "order", ""))
        logger.info("MT5 下单成功 %s %s %s lots order=%s", symbol, side, lots, ticket)
        return ticket

    def close_position(self, position_id: str, *, reason: str) -> None:
        """按持仓 ticket 平仓（hedging 账户必须带 ``position`` 字段）。"""
        mt5 = self._require_mt5()
        ticket = int(position_id.split(":")[-1])
        found = mt5.positions_get(ticket=ticket)
        if not found:
            raise BrokerUnavailableError(
                f"持仓不存在（券商侧）: ticket={ticket}",
                context={"position_id": position_id},
            )
        pos = found[0]
        is_long = int(pos.type) == int(mt5.POSITION_TYPE_BUY)
        symbol = str(pos.symbol)
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise BrokerUnavailableError(f"无法获取 {symbol} 行情", context={"symbol": symbol})
        req: dict[str, Any] = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(pos.volume),
            "type": mt5.ORDER_TYPE_SELL if is_long else mt5.ORDER_TYPE_BUY,
            "position": ticket,  # hedging 平仓的关键字段
            "price": float(tick.bid if is_long else tick.ask),
            "deviation": self.DEVIATION_POINTS,
            "magic": int(pos.magic),
            "comment": f"close:{reason}"[:31],
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._filling_mode(symbol),
        }
        result = mt5.order_send(req)
        retcode = int(getattr(result, "retcode", -1)) if result is not None else -1
        if retcode != int(mt5.TRADE_RETCODE_DONE):
            raise BrokerUnavailableError(
                f"平仓被拒: ticket={ticket} retcode={retcode}",
                context={"position_id": position_id, "retcode": retcode},
            )
        logger.warning("MT5 平仓成功 ticket=%s（%s）", ticket, reason)

    def positions(self) -> list[dict[str, Any]]:
        """当前持仓快照（券商侧；含全部 magic，运行器按 magic 域过滤）。"""
        mt5 = self._require_mt5()
        rows = mt5.positions_get() or []
        out: list[dict[str, Any]] = []
        for pos in rows:
            is_long = int(pos.type) == int(mt5.POSITION_TYPE_BUY)
            out.append(
                {
                    "position_id": str(pos.ticket),
                    "ticket": int(pos.ticket),
                    "symbol": str(pos.symbol),
                    "side": "long" if is_long else "short",
                    "lots": float(pos.volume),
                    "avg_price": float(pos.price_open),
                    "magic": int(pos.magic),
                }
            )
        return out

    def fills(self, since_ts: int) -> list[dict[str, Any]]:
        """``since_ts`` 之后的成交回执（``history_deals_get``，对账输入）。"""
        mt5 = self._require_mt5()
        date_from = datetime.fromtimestamp(since_ts, tz=UTC)
        date_to = datetime.fromtimestamp(int(self._now_fn()) + 60, tz=UTC)
        deals = mt5.history_deals_get(date_from, date_to) or []
        out: list[dict[str, Any]] = []
        for d in deals:
            out.append(
                {
                    "deal_id": int(d.ticket),
                    "order_id": str(d.order),
                    "symbol": str(d.symbol),
                    "side": "buy" if int(d.type) == 0 else "sell",  # DEAL_TYPE_BUY=0
                    "lots": float(d.volume),
                    "price": float(d.price),
                    "fee": float(getattr(d, "commission", 0.0) or 0.0)
                    + float(getattr(d, "swap", 0.0) or 0.0),
                    "profit": float(getattr(d, "profit", 0.0) or 0.0),
                    "ts": int(d.time),
                    "magic": int(d.magic),
                    "reason": "mt5-deal",
                }
            )
        return out

    def flatten_all(self, *, reason: str) -> None:
        """紧急全平——**账户全部持仓**（含人工/其他 EA 的仓，慎用）。

        实盘运行器通过 magic 域包装（只平自己开的仓）调用逐仓
        :meth:`close_position`；本方法保留端口契约的"账户级全平"语义，
        供 kill 开关的最后手段使用。
        """
        mt5 = self._require_mt5()
        for pos in mt5.positions_get() or []:
            try:
                self.close_position(str(pos.ticket), reason=reason)
            except BrokerUnavailableError as exc:  # noqa: PERF203 - 尽力而为逐仓平
                logger.error("急停平仓失败 ticket=%s: %s", pos.ticket, exc)

    def symbol_spec(self, symbol: str) -> BrokerSpec:
        """从终端读品种规格（tick_value / lot_step / min_lot / contract_size）。"""
        mt5 = self._require_mt5()
        info = mt5.symbol_info(symbol)
        if info is None:
            raise BrokerUnavailableError(
                f"品种不存在（券商侧）: {symbol}", context={"symbol": symbol}
            )
        return BrokerSpec(
            tick_value=float(info.trade_tick_value or 0.0) or 1.0,
            lot_step=float(info.volume_step),
            min_lot=float(info.volume_min),
            contract_size=float(info.trade_contract_size),
        )

    def account_summary(self) -> dict[str, Any]:
        """账户摘要（只读；状态页与启动日志用，不下单）。"""
        mt5 = self._require_mt5()
        info = mt5.account_info()
        if info is None:
            raise BrokerUnavailableError("无法读取账户信息（终端未登录?）")
        return {
            "login": int(info.login),
            "server": str(info.server),
            "currency": str(info.currency),
            "balance": float(info.balance),
            "equity": float(info.equity),
            "leverage": int(info.leverage),
        }

    # ── 内部 ─────────────────────────────────────────────────────────────

    def _filling_mode(self, symbol: str) -> int:
        """按品种允许的成交模式选 filling（FOK 优先，其次 IOC，兜底 RETURN）。"""
        mt5 = self._require_mt5()
        info = mt5.symbol_info(symbol)
        modes = int(getattr(info, "filling_mode", 0)) if info is not None else 0
        # SYMBOL_FILLING_FOK=1, SYMBOL_FILLING_IOC=2（位标志）
        if modes & 1:
            return int(mt5.ORDER_FILLING_FOK)
        if modes & 2:
            return int(mt5.ORDER_FILLING_IOC)
        return int(mt5.ORDER_FILLING_RETURN)
