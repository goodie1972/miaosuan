# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for full license text.

"""实盘运行回路（路线图 R4）：MT5Broker + 熔断 + magic 域对账。

与 :class:`~miaosuan.runtime.paper.PaperRunner` 的差异：

* broker = MT5Broker 的 **magic 域包装**（只看见/只平自己开的仓——
  账户里人工或其他 EA 的持仓不受本运行器影响，对账也只对本域）；
* 成交即时回报，无本地撮合（``on_bar`` 空实现）；
* **熔断接线**（R4.3）：连续拒单 / 单日最大亏损 / 极端滑点 → 任一命中
  即「全平（本域）+ 创建 STOP 文件（急停持久化）+ 告警 + 停机」；
* 对账增强（R4.2）：回执以 ``history_deals_get`` 为准逐笔比对，
  不一致**先平后报**（继承启动/周期对账流程）。

安全边界：本模块绝不自行启动——``CLI live start`` 必须显式
``--i-understand-real-money``；凭证经构造注入（env 边界在 CLI）。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import (
    BrokerKillswitchError,
    BrokerReconcileError,
    BrokerUnavailableError,
    MiaoSuanError,
)
from .breaker import BreakerConfig, CircuitBreaker
from .brokers.mt5 import MT5Broker
from .paper import PaperConfig, PaperRunner
from .ports import Bar, ExecutionPort, NotifierPort
from .reconcile import reconcile
from .scheduler import WatchItem
from .state import kill_file_path

__all__ = ["LiveConfig", "LiveRunner", "magic_from_spec"]

logger = logging.getLogger(__name__)


def magic_from_spec(spec_id: str) -> int:
    """spec_id → 稳定魔法号（1..2^31-1；同一策略恒定，券商侧按此归组）。"""
    return int(spec_id[:8], 16) % 2_147_483_646 + 1


class _MagicScopedBroker(ExecutionPort):
    """magic 域视图：positions/fills/flatten 只作用于本运行器开的仓。"""

    def __init__(self, inner: MT5Broker, magic: int) -> None:
        self._inner = inner
        self._magic = int(magic)

    def submit(self, symbol: str, side: str, lots: float, *, magic: int, comment: str = "") -> str:
        if int(magic) != self._magic:
            raise MiaoSuanError(f"magic 不属于本运行器域: {magic}（期望 {self._magic}）")
        return self._inner.submit(symbol, side, lots, magic=magic, comment=comment)

    def close_position(self, position_id: str, *, reason: str) -> None:
        self._inner.close_position(position_id, reason=reason)

    def positions(self) -> list[dict[str, Any]]:
        return [p for p in self._inner.positions() if int(p["magic"]) == self._magic]

    def fills(self, since_ts: int) -> list[dict[str, Any]]:
        return [f for f in self._inner.fills(since_ts) if int(f["magic"]) == self._magic]

    def flatten_all(self, *, reason: str) -> None:
        """仅平**本域**持仓（逐仓 close_position；不碰账户其他仓位）。"""
        for pos in self.positions():
            self._inner.close_position(str(pos["position_id"]), reason=reason)

    def on_bar(self, bar: Bar) -> list[dict[str, Any]]:
        return self._inner.on_bar(bar)  # MT5Broker 默认空实现（真实成交即时回报）


@dataclass(frozen=True)
class LiveConfig(PaperConfig):
    """实盘参数 = 纸面参数 + 熔断阈值。"""

    breaker: BreakerConfig = BreakerConfig()


class LiveRunner(PaperRunner):
    """实盘运行器：PaperRunner 骨架 + MT5Broker + 熔断 + magic 域对账。"""

    def __init__(
        self,
        item: WatchItem,
        port: Any,
        *,
        profile_name: str,
        config: LiveConfig,
        broker: MT5Broker,
        state_root: Path | None = None,
        notifier: NotifierPort | None = None,
    ) -> None:
        self.raw_broker = broker
        self.magic = magic_from_spec(item.engine.spec.spec_id)
        scoped = _MagicScopedBroker(broker, self.magic)
        super().__init__(
            item,
            port,
            profile_name=profile_name,
            config=config,
            state_root=state_root,
            notifier=notifier,
            broker=scoped,
        )
        self.breaker = CircuitBreaker(config.breaker)
        self._requested_prices: dict[int, float] = {}
        self._fills_watermark = int(time.time()) - 86400  # 近 24h 回执窗口

    # ── 生命周期 ─────────────────────────────────────────────────────────

    def start(self) -> None:
        """实盘启动链（D6 + R4.4）：kill 检查 → 锁 → 终端连接 → 恢复 → 对账。"""
        self.raw_broker.connect()  # 先建通道：后续对账需要真实持仓/回执
        try:
            super().start()  # kill 文件 → 锁 → 账本恢复 → 启动对账（magic 域）
        except MiaoSuanError:
            self.raw_broker.close()
            raise
        logger.warning(
            "实盘模式就绪 %s（magic=%d, 账户=%s）",
            self.item.label,
            self.magic,
            self.raw_broker.account_summary(),
        )

    def stop(self) -> None:
        self.raw_broker.close()
        super().stop()

    # ── 运行回路 ─────────────────────────────────────────────────────────

    def _on_bar(self, item: WatchItem, bar: Bar, position: float, prev_position: float) -> None:
        """一根收盘 bar：回执核查 → 信号 → 调仓 → 对账（顺序固定）。"""
        del prev_position
        self._remember(bar)
        self.broker.on_bar(bar)  # MT5：空实现（即时成交）

        # 1) 回执核查（熔断：日亏 / 滑点；回执窗口从上次水位推进）
        self.breaker.check_fills(
            self.broker.fills(self._fills_watermark),
            requested_prices=self._requested_prices,
        )
        self._fills_watermark = int(time.time()) - 60
        if self.breaker.tripped:
            self._trip_flow()
            return

        # 2) 信号 → 风险手数 → 求差调仓（拒单计数在调仓里推进）
        signal = item.engine.last_position
        target_lots = self.policy.target_lots(
            signal=signal,
            high=np.asarray(self._high, dtype=np.float64),
            low=np.asarray(self._low, dtype=np.float64),
            close=np.asarray(self._close, dtype=np.float64),
            broker_spec=self.profile.broker_spec,  # type: ignore[arg-type]
        )
        self._rebalance_safe(target_lots)
        if self.breaker.tripped:
            self._trip_flow()
            return

        # 3) 对账（R4.2：不一致先平后报）
        try:
            reconcile(self.ledger, self.broker)
        except BrokerReconcileError:
            logger.warning("实盘对账失败——全平告警")
            self.broker.flatten_all(reason="reconcile-fail")
            if self.notifier is not None:
                self.notifier.notify(
                    title=f"实盘对账失败 {self.item.label}",
                    body="账本与券商回执存在差异，已全平待人工介入",
                    level="error",
                )
            raise

    # ── 内部 ─────────────────────────────────────────────────────────────

    def _rebalance_safe(self, target_lots: float) -> None:
        """调仓 + 拒单熔断计数：拒单连续达上限即熔断（先平后停机）。"""
        spec = self.profile.broker_spec
        assert spec is not None
        symbol = self.item.symbol
        current = 0.0
        for pos in self.broker.positions():
            if pos["symbol"] == symbol:
                current = float(pos["lots"]) * (1.0 if pos["side"] == "long" else -1.0)
        delta = round(target_lots - current, 8)
        if abs(delta) < float(spec.min_lot) - 1e-9:
            self.breaker.on_order_accepted()  # 无需调仓不算失败
            return
        side = "buy" if delta > 0 else "sell"
        try:
            ticket = self.broker.submit(
                symbol, side, abs(delta), magic=self.magic, comment="signal"
            )
        except BrokerUnavailableError as exc:
            self.breaker.on_order_rejected(str(exc))
            logger.warning("下单被拒（连续 %d 次）: %s", self.breaker.consecutive_rejections, exc)
            return
        self.breaker.on_order_accepted()
        self._requested_prices[int(ticket)] = self._last_request_price(side, symbol)

    def _last_request_price(self, side: str, symbol: str) -> float:
        """记录请求价（滑点核查对照；实现取自最近一次 tick 快照）。"""
        price = getattr(self.raw_broker, "last_request_price", {}).get(symbol)
        return float(price) if price else 0.0

    def _trip_flow(self) -> None:
        """熔断落地：全平（本域）→ 创建 STOP（急停持久化）→ 告警 → 停机。"""
        reason = self.breaker.reason or "circuit-breaker"
        logger.warning("熔断触发：%s —— 全平停机", reason)
        try:
            self.broker.flatten_all(reason=f"breaker:{reason}")
        except BrokerUnavailableError as exc:  # noqa: BLE001 - 平仓失败也必须停机
            logger.error("熔断全平失败（人工必须介入）：%s", exc)
        kill = kill_file_path(self.state_root)
        kill.parent.mkdir(parents=True, exist_ok=True)
        kill.write_text(f"breaker:{reason}\n", encoding="utf-8")
        if self.notifier is not None:
            self.notifier.notify(
                title=f"实盘熔断 {self.item.label}",
                body=f"{reason}：已全平并创建 STOP 文件，人工确认后方可重启",
                level="error",
            )
        raise BrokerKillswitchError(f"熔断触发：{reason}")
