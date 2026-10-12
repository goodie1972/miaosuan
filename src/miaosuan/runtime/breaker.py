# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for full license text.

"""熔断器（路线图 R4.3）：单日最大亏损 / 连续拒单 / 极端滑点。

任一阈值命中 → ``tripped`` 置位并携带原因；运行器职责：全平
（``flatten_all``）+ 创建 STOP 文件（急停持久化）+ 告警 + 抛
:class:`~miaosuan.errors.BrokerKillswitchError`。本模块只做**判定**，
不做 IO——便于单测；STOP/平仓在 :mod:`~miaosuan.runtime.live`。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

__all__ = ["BreakerConfig", "CircuitBreaker"]


@dataclass(frozen=True)
class BreakerConfig:
    """熔断阈值（实盘默认从严；0 = 关闭该项）。"""

    #: 单日已实现亏损下限（USD，含手续费/库存费；≤ 该值触发）。0 = 关闭。
    max_daily_loss_usd: float = 100.0
    #: 连续拒单次数上限（每次成功下单清零）。0 = 关闭。
    max_consecutive_rejections: int = 3
    #: 成交价相对请求价的最大偏差（比率，如 0.005 = 0.5%）。0 = 关闭。
    max_slippage_ratio: float = 0.005


@dataclass
class CircuitBreaker:
    """实盘熔断判定（状态机：拒单计数 + 当日已实现盈亏 + 滑点核查）。"""

    config: BreakerConfig = field(default_factory=BreakerConfig)
    consecutive_rejections: int = 0
    tripped: bool = False
    reason: str = ""

    # ── 事件入口 ─────────────────────────────────────────────────────────

    def on_order_rejected(self, detail: str) -> None:
        """拒单计数；达上限即熔断。"""
        if self.tripped:
            return
        self.consecutive_rejections += 1
        limit = self.config.max_consecutive_rejections
        if limit and self.consecutive_rejections >= limit:
            self._trip(f"连续拒单 {self.consecutive_rejections} 次（上限 {limit}）：{detail}")

    def on_order_accepted(self) -> None:
        """成功下单清零拒单计数。"""
        self.consecutive_rejections = 0

    def check_fills(
        self,
        fills: list[dict[str, float | int | str]],
        *,
        requested_prices: dict[int, float] | None = None,
    ) -> None:
        """核查成交回执：当日已实现盈亏 + 滑点（可带请求价对照）。

        :param fills: ``broker.fills(since_ts)`` 形态（price/lots/side/profit/fee/ts）。
        :param requested_prices: order ticket → 请求价（滑点核查；可省略）。
        """
        if self.tripped:
            return
        if self.config.max_daily_loss_usd:
            loss = self._daily_realized_pnl(fills)
            if loss <= -abs(self.config.max_daily_loss_usd):
                self._trip(
                    f"单日已实现亏损 {loss:.2f} USD（阈值 -{abs(self.config.max_daily_loss_usd):.2f}）"
                )
                return
        if self.config.max_slippage_ratio and requested_prices:
            for fill in fills:
                order_id = str(fill.get("order_id", ""))
                requested = requested_prices.get(int(order_id)) if order_id.isdigit() else None
                if requested is None or not requested:
                    continue
                slip = abs(float(fill["price"]) - requested) / requested
                if slip > self.config.max_slippage_ratio:
                    self._trip(
                        f"极端滑点 {slip:.4%}（阈值 {self.config.max_slippage_ratio:.4%}）"
                        f" deal={fill.get('deal_id')}"
                    )
                    return

    # ── 查询 ─────────────────────────────────────────────────────────────

    @staticmethod
    def _daily_realized_pnl(fills: list[dict[str, float | int | str]]) -> float:
        """今日已实现盈亏（profit + fee；以 UTC 日界聚合）。"""
        today = datetime.now(tz=UTC).date()
        total = 0.0
        for fill in fills:
            ts = int(fill.get("ts", 0))
            if datetime.fromtimestamp(ts, tz=UTC).date() != today:
                continue
            total += float(fill.get("profit", 0.0)) + float(fill.get("fee", 0.0))
        return round(total, 2)

    # ── 内部 ─────────────────────────────────────────────────────────────

    def _trip(self, reason: str) -> None:
        self.tripped = True
        self.reason = reason
