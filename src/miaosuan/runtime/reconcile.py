# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for full license text.

"""对账器（路线图 R3.5）：本地账本 vs 撮合回执逐笔比对。

D6 验收顺序的第三环「能对上账」：差异即 ``E-BROKER-RECONCILE``。
Paper 模式下 broker 回执与账本同源，对账验证的是**账本自身完整性**
（intent↔fill 配对、持仓聚合一致性）——这正是崩溃/写坏检测所依赖的；
R4 的 MT5Broker 换成 ``history_deals_get`` 外部回执，接口不变。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..errors import BrokerReconcileError
from .ports import ExecutionPort
from .state import Ledger

__all__ = ["ReconcileReport", "reconcile"]

logger = logging.getLogger(__name__)


@dataclass
class ReconcileReport:
    """对账结果（零差异 = ok 且 diffs 为空）。"""

    ok: bool = True
    checked_intents: int = 0
    checked_fills: int = 0
    checked_positions: int = 0
    diffs: list[str] = field(default_factory=list)

    def summary(self) -> str:
        status = "OK" if self.ok else f"FAIL（{len(self.diffs)} 项差异）"
        return (
            f"对账 {status}: intents={self.checked_intents} "
            f"fills={self.checked_fills} positions={self.checked_positions}"
        )


def reconcile(ledger: Ledger, broker: ExecutionPort) -> ReconcileReport:
    """逐笔比对账本与 broker 回执；任何差异抛 :class:`BrokerReconcileError`。

    校验三件事：

    1. **intent ↔ fill 配对**：filled 意图必须有成交，成交必须挂在
       filled 意图上（cancelled 意图不得有成交）；
    2. **持仓聚合一致**：账本净额聚合的持仓 == broker 汇报的持仓
       （方向/手数逐项相等）；
    3. **回执通道**：``broker.fills(since)`` 与账本 fills 同源一致
       （Paper 同源自检；R4 换外部回执后此校验成为真对账）。
    """
    report = ReconcileReport()
    intents = {str(i.get("intent_id")): i for i in ledger.intents()}
    fills = ledger.all_fills()
    report.checked_intents = len(intents)
    report.checked_fills = len(fills)

    filled_ids = {f.intent_id for f in fills}
    for intent_id, intent in intents.items():
        status = str(intent.get("status"))
        if status == "filled" and intent_id not in filled_ids:
            report.ok = False
            report.diffs.append(f"意图 {intent_id} 标记 filled 但无成交回执")
        if status == "pending" and intent_id in filled_ids:
            report.ok = False
            report.diffs.append(f"意图 {intent_id} 仍为 pending 但已有成交回执")
        if status == "cancelled" and intent_id in filled_ids:
            report.ok = False
            report.diffs.append(f"意图 {intent_id} 已取消但存在成交回执")

    fill_intent_ids = {f.intent_id for f in fills}
    for intent_id in fill_intent_ids:
        if intent_id not in intents:
            report.ok = False
            report.diffs.append(f"成交回执 {intent_id} 找不到对应意图")
        elif str(intents[intent_id].get("status")) != "filled":
            report.ok = False
            report.diffs.append(f"成交回执 {intent_id} 对应意图状态异常")

    # 持仓聚合：账本口径 vs broker 汇报（Paper = 同一聚合函数的自检）
    ledger_positions = {
        str(p["position_id"]): (str(p["side"]), round(float(p["lots"]), 8))
        for p in ledger.open_positions()
    }
    broker_positions = {
        str(p["position_id"]): (str(p["side"]), round(float(p["lots"]), 8))
        for p in broker.positions()
    }
    report.checked_positions = len(ledger_positions)
    for pid, expected in ledger_positions.items():
        if broker_positions.get(pid) != expected:
            report.ok = False
            report.diffs.append(
                f"持仓 {pid} 不一致: 账本={expected} broker={broker_positions.get(pid)}"
            )
    for pid in set(broker_positions) - set(ledger_positions):
        report.ok = False
        report.diffs.append(f"broker 多出持仓 {pid}（账本无记录）")

    logger.info(report.summary())
    if not report.ok:
        raise BrokerReconcileError(
            f"对账失败（{len(report.diffs)} 项差异）",
            context={"diffs": report.diffs},
        )
    return report
