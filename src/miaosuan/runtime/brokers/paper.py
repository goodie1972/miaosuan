# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for full license text.

"""PaperBroker（路线图 R3.2）：纸面撮合与成本口径的锚点。

事件式撮合（Sample C v1.1）：``submit`` 只入队（账本记 pending intent），
**下一根已收盘 bar 的开盘价**成交——submit 时刻下一根 bar 尚不存在，
"下一根开盘"只能由调度回路在 bar 到来时回调 :meth:`on_bar` 完成。

口径（单一事实，禁止本文件出现任何硬编码费率）：

* 滑点/佣金取自 ``profile.cost_model``（CostModel，单边比率，买卖可不对称）；
* 合约规格（contract_size / lot_step / min_lot）取自 ``profile.broker_spec``；
* ``flatten_all`` 是急停路径的**明示例外**：按当前 mark 价（最近收盘价）
  即时成交，不等下一根开盘——kill ≤ 1 个调度周期内生效的验收依赖它。

崩溃恢复：构造时把账本中的 pending intents 重放回撮合队列（D6「能恢复」）。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ...errors import MiaoSuanError
from ...market.profiles import FrozenMarketProfile
from ..ports import Bar, ExecutionPort
from ..state import Ledger

__all__ = ["PaperBroker"]

logger = logging.getLogger(__name__)

_SIDES = ("buy", "sell")
_LOT_EPS = 1e-9


@dataclass
class _QueuedIntent:
    """撮合队列条目（账本 pending intent 的内存镜像）。"""

    intent_id: str
    symbol: str
    side: str
    lots: float
    magic: int
    comment: str


class PaperBroker(ExecutionPort):
    """纸面 broker：意图入队 → 下一根开盘价 ± 滑点成交，佣金入账。"""

    def __init__(
        self,
        profile: FrozenMarketProfile,
        ledger: Ledger,
        *,
        now_fn: Callable[[], float] = time.time,
    ) -> None:
        if profile.broker_spec is None:
            raise MiaoSuanError(
                f"画像 {profile.name!r} 未携带 broker_spec——纸面/实盘执行需要"
                "经纪商品种规格（lot_step/min_lot/contract_size）",
                context={"profile": profile.name},
            )
        self._profile = profile
        self._spec = profile.broker_spec
        self._cost = profile.cost_model
        self._ledger = ledger
        self._now_fn = now_fn
        self._pending: dict[str, list[_QueuedIntent]] = {}
        self._marks: dict[str, float] = {}
        # 崩溃恢复：账本里遗留的 pending intents 重放回队列（D6 能恢复）
        for intent in ledger.pending_intents():
            self._pending.setdefault(str(intent["symbol"]), []).append(
                _QueuedIntent(
                    intent_id=str(intent["intent_id"]),
                    symbol=str(intent["symbol"]),
                    side=str(intent["side"]),
                    lots=float(intent["lots"]),
                    magic=int(intent["magic"]),
                    comment=str(intent.get("comment", "")),
                )
            )
        if self._pending:
            logger.warning(
                "账本恢复：重放 %d 条未完成意图（上个实例崩溃遗留）",
                sum(len(v) for v in self._pending.values()),
            )

    # ── ExecutionPort ─────────────────────────────────────────────────────

    def submit(self, symbol: str, side: str, lots: float, *, magic: int, comment: str = "") -> str:
        """校验并登记挂单意图（未成交），返回 intent_id。"""
        if side not in _SIDES:
            raise MiaoSuanError(f"非法方向: {side!r}", context={"allowed": list(_SIDES)})
        lots = round(float(lots), 8)
        if lots <= 0:
            raise MiaoSuanError("手数必须为正", context={"lots": lots})
        if lots < float(self._spec.min_lot) - _LOT_EPS:
            raise MiaoSuanError(
                "手数低于最小手数（不交易，禁止向上取整）",
                context={"lots": lots, "min_lot": float(self._spec.min_lot)},
            )
        steps = round(lots / float(self._spec.lot_step))
        if abs(steps * float(self._spec.lot_step) - lots) > _LOT_EPS:
            raise MiaoSuanError(
                "手数未按 lot_step 对齐",
                context={"lots": lots, "lot_step": float(self._spec.lot_step)},
            )
        intent_id = self._ledger.append_intent(
            symbol, side, lots, magic=int(magic), comment=comment
        )
        self._pending.setdefault(symbol, []).append(
            _QueuedIntent(intent_id, symbol, side, lots, int(magic), comment)
        )
        logger.info("挂单 %s %s %s %s lots（待下一根开盘成交）", intent_id, symbol, side, lots)
        return intent_id

    def close_position(self, position_id: str, *, reason: str) -> None:
        """按持仓 id 平仓——登记反向意图，按撮合规则下一根开盘成交。"""
        target = self._find_position(position_id)
        if target is None:
            raise MiaoSuanError(
                f"持仓不存在: {position_id}",
                context={"open": [p["position_id"] for p in self.positions()]},
            )
        offset_side = "sell" if target["side"] == "long" else "buy"
        self.submit(
            str(target["symbol"]),
            offset_side,
            float(target["lots"]),
            magic=int(target["magic"]),
            comment=f"close:{reason}",
        )

    def positions(self) -> list[dict[str, Any]]:
        return self._ledger.open_positions()

    def fills(self, since_ts: int) -> list[dict[str, Any]]:
        return [f.to_dict() for f in self._ledger.fills(since_ts)]

    def flatten_all(self, *, reason: str) -> None:
        """紧急全平（急停例外：mark 价即时成交）+ 撤销未成交意图。"""
        for queue in self._pending.values():
            for intent in queue:
                self._ledger.mark_intent_status(intent.intent_id, "cancelled")
            queue.clear()
        for pos in self.positions():
            symbol = str(pos["symbol"])
            mark = self._marks.get(symbol)
            if mark is None:
                raise MiaoSuanError(
                    f"无 {symbol} 的 mark 价（尚未见过收盘 bar），无法即时平仓",
                    context={"position_id": pos["position_id"]},
                )
            offset_side = "sell" if pos["side"] == "long" else "buy"
            lots = float(pos["lots"])
            price = (
                mark * (1 - self._cost.slippage)
                if offset_side == "sell"
                else mark * (1 + self._cost.slippage)
            )
            fee = price * lots * float(self._spec.contract_size) * self._cost.commission
            intent_id = self._ledger.append_intent(
                symbol,
                offset_side,
                lots,
                magic=int(pos["magic"]),
                comment=f"flatten:{reason}",
            )
            self._ledger.append_fill(
                intent_id,
                symbol,
                offset_side,
                lots,
                price,
                fee,
                magic=int(pos["magic"]),
                reason=f"flatten:{reason}",
                ts=int(self._now_fn()),
            )
            self._ledger.mark_intent_status(intent_id, "filled")
            logger.warning(
                "急停全平 %s %s %s lots @ %.4f（%s）", symbol, offset_side, lots, price, reason
            )

    # ── 撮合回路（调度器每根收盘 bar 回调）────────────────────────────────

    def on_bar(self, bar: Bar) -> list[dict[str, Any]]:
        """一根 bar 的两件事：先按**本根开盘价**撮合上一根收盘时的挂单，
        再把 mark 推进到本根收盘价。返回本根产生的成交回执。"""
        filled: list[dict[str, Any]] = []
        queue = self._pending.get(bar.symbol, [])
        for intent in queue:
            price = bar.open * (
                (1 + self._cost.slippage) if intent.side == "buy" else (1 - self._cost.slippage)
            )
            fee = price * intent.lots * float(self._spec.contract_size) * self._cost.commission
            fill_id = self._ledger.append_fill(
                intent.intent_id,
                bar.symbol,
                intent.side,
                intent.lots,
                price,
                fee,
                magic=intent.magic,
                reason=intent.comment or "signal",
                ts=int(self._now_fn()),
            )
            self._ledger.mark_intent_status(intent.intent_id, "filled")
            fill = {
                "fill_id": fill_id,
                "intent_id": intent.intent_id,
                "symbol": bar.symbol,
                "side": intent.side,
                "lots": intent.lots,
                "price": price,
                "fee": fee,
                "magic": intent.magic,
                "reason": intent.comment or "signal",
            }
            filled.append(fill)
            logger.info(
                "成交 %s %s %s lots @ %.4f fee=%.4f（下一根开盘口径）",
                fill_id,
                bar.symbol,
                intent.side,
                price,
                fee,
            )
        queue.clear()
        self._marks[bar.symbol] = bar.close
        return filled

    # ── 内部 ─────────────────────────────────────────────────────────────

    def _find_position(self, position_id: str) -> dict[str, Any] | None:
        for pos in self.positions():
            if pos["position_id"] == position_id:
                return pos
        return None

    @property
    def pending_count(self) -> int:
        """未成交意图数（观测用）。"""
        return sum(len(v) for v in self._pending.values())
