# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for full license text.

"""纸面运行回路（路线图 R3）：信号 → 风险手数 → 撮合 → 对账 → kill。

D6 验收顺序的实现（缺一不得进入实盘，R4 复用本模块的顺序）::

    启动：门禁/词表安全链（watch add / 装配时）→ kill 文件检查
          → 单实例锁 → 账本加载 + 崩溃恢复（pending 重放）→ 启动对账先行
    运行：每根收盘 bar —— 撮合上一轮挂单（下一根开盘口径）→ 引擎出信号
          → 风险手数 → 与当前持仓求差 → 调仓（submit / close_position）
          → 对账（零差异是常态，差异即平仓告警）
    周期：kill 文件检查（≤ 1 个调度周期生效，急停例外按 mark 即时平仓）
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..errors import BrokerKillswitchError, BrokerReconcileError, MiaoSuanError
from ..market.profiles import get_profile
from .brokers.paper import PaperBroker
from .ports import Bar, DataPort, ExecutionPort, NotifierPort
from .reconcile import reconcile
from .risk import VolTargetPolicy
from .scheduler import CloseScheduler, WatchItem
from .state import Ledger, LockHandle, kill_file_exists, state_dir

__all__ = ["PaperConfig", "PaperRunner", "paper_ledger_path"]

logger = logging.getLogger(__name__)


def paper_ledger_path(symbol: str, timeframe: str, root: Path | None = None) -> Path:
    """纸面账本路径（每 watch 条目一本：``paper_{symbol}_{tf}.json``）。"""
    return state_dir(root) / f"paper_{symbol}_{timeframe}.json"


@dataclass(frozen=True)
class PaperConfig:
    """纸面模式参数（风险预算 + 轮询节奏）。"""

    risk_usd_per_bar: float = 200.0
    atr_window: int = 14
    sharpe_weight: float = 1.0
    w_min: float = 0.5
    w_max: float = 2.0
    poll_seconds: float = 30.0


class PaperRunner:
    """单条目的纸面运行器：引擎 + 风险 + broker + 账本 + 对账 + kill。

    :param item: watch 条目（品种/周期/引擎）。
    :param port: 数据端口（交给调度器轮询；本类不直接消费）。
    :param profile_name: 市场画像名（须携带 broker_spec 与成本模型）。
    """

    def __init__(
        self,
        item: WatchItem,
        port: DataPort,
        *,
        profile_name: str,
        config: PaperConfig,
        state_root: Path | None = None,
        notifier: NotifierPort | None = None,
        broker: ExecutionPort | None = None,
    ) -> None:
        profile = get_profile(profile_name)
        if profile.broker_spec is None:
            raise MiaoSuanError(
                f"画像 {profile_name!r} 未携带 broker_spec，无法纸面执行",
                context={"profile": profile_name},
            )
        self.item = item
        self.port = port
        self.profile = profile
        self.config = config
        self.state_root = state_root
        self.notifier = notifier
        self.ledger = Ledger(paper_ledger_path(item.symbol, item.timeframe, state_root))
        self.lock = LockHandle(state_dir(state_root) / f"paper_{item.symbol}_{item.timeframe}.lock")
        # broker 可注入（LiveRunner 传 MT5Broker 的 magic 域包装）；缺省纸面撮合
        self.broker: ExecutionPort = (
            broker if broker is not None else PaperBroker(profile, self.ledger)
        )
        self.policy = VolTargetPolicy(
            risk_usd_per_bar=config.risk_usd_per_bar,
            atr_window=config.atr_window,
            sharpe_weight=config.sharpe_weight,
            w_min=config.w_min,
            w_max=config.w_max,
        )
        self._high: list[float] = []
        self._low: list[float] = []
        self._close: list[float] = []

    # ── 生命周期 ─────────────────────────────────────────────────────────

    def start(self) -> None:
        """启动安全链（顺序不可调换）：kill 检查 → 锁 → 恢复 → 对账先行。"""
        if kill_file_exists(self.state_root):
            raise BrokerKillswitchError("检测到 STOP 文件，纸面模式拒绝启动")
        self.lock.acquire()  # E-RUNTIME-STATE-LOCK
        try:
            reconcile(self.ledger, self.broker)  # 启动对账先行（账本恢复后自检）
        except BrokerReconcileError:
            logger.warning("启动对账失败，执行全平后重抛")
            self.broker.flatten_all(reason="reconcile-fail-at-start")
            self.lock.release()
            raise
        logger.info(
            "纸面模式就绪 %s（profile=%s, 恢复 fills=%d, pending=%d）",
            self.item.label,
            self.profile.name,
            len(self.ledger.all_fills()),
            getattr(self.broker, "pending_count", 0),
        )

    def stop(self) -> None:
        """释放锁（正常退出路径）。"""
        self.lock.release()

    # ── 运行回路 ─────────────────────────────────────────────────────────

    def run_with_scheduler(
        self,
        scheduler: CloseScheduler,
        *,
        once: bool = False,
        stop_after_seconds: float | None = None,
    ) -> None:
        """复用收盘调度器：kill 检查挂 on_cycle，交易逻辑挂 on_bar。"""
        scheduler.run(
            [self.item],
            self._on_bar,
            once=once,
            stop_after_seconds=stop_after_seconds,
            on_cycle=self._check_kill,
        )

    def _check_kill(self) -> None:
        """kill 文件检查（每周期一次，与是否有新 bar 无关——≤1 周期生效）。"""
        if kill_file_exists(self.state_root):
            logger.warning("kill 文件命中——全平停机")
            self.broker.flatten_all(reason="kill-file")
            if self.notifier is not None:
                self.notifier.notify(
                    title=f"纸面急停 {self.item.label}",
                    body="STOP 文件命中：已全平并停机",
                    level="warning",
                )
            raise BrokerKillswitchError("STOP 文件命中，纸面模式已平仓停机")

    def _on_bar(self, item: WatchItem, bar: Bar, position: float, prev_position: float) -> None:
        """一根收盘 bar：撮合 → 信号 → 调仓 → 对账（顺序固定）。"""
        del prev_position
        self._remember(bar)
        self.broker.on_bar(bar)  # 1) 上一轮挂单按本根开盘价成交
        signal = item.engine.last_position  # == position（调度器喂引擎后的现值）
        target_lots = self.policy.target_lots(
            signal=signal,
            high=np.asarray(self._high, dtype=np.float64),
            low=np.asarray(self._low, dtype=np.float64),
            close=np.asarray(self._close, dtype=np.float64),
            broker_spec=self.profile.broker_spec,  # type: ignore[arg-type]
        )
        self._rebalance(target_lots)  # 2) 求差调仓
        try:
            reconcile(self.ledger, self.broker)  # 3) 对账（零差异是常态）
        except BrokerReconcileError:
            logger.warning("对账失败——全平告警")
            self.broker.flatten_all(reason="reconcile-fail")
            if self.notifier is not None:
                self.notifier.notify(
                    title=f"纸面对账失败 {self.item.label}",
                    body="账本与回执存在差异，已全平待人工介入",
                    level="error",
                )
            raise

    # ── 内部 ─────────────────────────────────────────────────────────────

    def _remember(self, bar: Bar) -> None:
        """维护 ATR 价格缓冲（与引擎历史同步推进）。"""
        self._high.append(bar.high)
        self._low.append(bar.low)
        self._close.append(bar.close)

    def _rebalance(self, target_lots: float) -> None:
        """持仓差额调仓：净持仓 → 目标手数（多正空负），差额按方向 submit。

        求差式调仓：目标与现持仓都是 lot_step 的倍数，差额天然对齐；
        差额低于 min_lot（含 0）不动——不交易的手数不产生订单。
        """
        spec = self.profile.broker_spec
        assert spec is not None  # __init__ 已校验
        symbol = self.item.symbol
        current = 0.0
        for pos in self.broker.positions():
            if pos["symbol"] == symbol:
                current = float(pos["lots"]) * (1.0 if pos["side"] == "long" else -1.0)
        delta = round(target_lots - current, 8)
        if abs(delta) < float(spec.min_lot) - 1e-9:
            return
        magic = int(self.item.engine.spec.spec_id[:8], 16) % 2_147_483_646 + 1
        side = "buy" if delta > 0 else "sell"
        self.broker.submit(symbol, side, abs(delta), magic=magic, comment="signal")
