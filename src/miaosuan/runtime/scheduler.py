# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for full license text.

"""收盘调度与 watchlist 运行回路（路线图 R2.1 / R2.5）。

职责边界：调度器只做「轮询 → 收盘判定 → 去重 → 分发」，不研究、不下单；
信号由 :class:`~miaosuan.runtime.signal_engine.SignalEngine` 产出，
执行与通知经端口注入（信号模式不下单，纸面/实盘在 R3/R4 接入）。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..errors import MiaoSuanError
from .ports import Bar, DataPort, NotifierPort
from .signal_engine import SignalEngine, engine_signal_summary

__all__ = ["WatchItem", "CloseScheduler", "log_bar_event", "run_watchlist", "timeframe_seconds"]

logger = logging.getLogger(__name__)

#: 各周期秒数（收盘判定用；与 fetchers 的口径一致，MN1 按 30 天近似）。
_TF_SECONDS: dict[str, int] = {
    "M1": 60,
    "M5": 300,
    "M15": 900,
    "M30": 1800,
    "H1": 3600,
    "H4": 14400,
    "D1": 86400,
    "W1": 604800,
    "MN1": 2_592_000,
}


def timeframe_seconds(timeframe: str) -> int:
    """周期标识 → 秒数（未知周期抛 :class:`MiaoSuanError`）。"""
    tf = (timeframe or "").upper()
    if tf not in _TF_SECONDS:
        raise MiaoSuanError(
            f"不支持的周期: {timeframe!r}",
            context={"supported": sorted(_TF_SECONDS)},
        )
    return _TF_SECONDS[tf]


@dataclass(frozen=True)
class WatchItem:
    """一个 watch 条目：品种 + 周期 + 已加载的信号引擎。"""

    symbol: str
    timeframe: str
    engine: SignalEngine

    @property
    def label(self) -> str:
        return f"{self.symbol}.{self.timeframe}"


class CloseScheduler:
    """收盘调度：轮询 DataPort → 按「周期已走完」判定收盘 → 水位去重。

    收盘判定与经纪商时钟无关地严格：``bar.ts + tf_seconds <= now_utc``
    才视为收盘（bar 区间已完整走完），未收盘 bar 永不下发——收盘纪律
    在调度层与端口层双重强制。断线补拉靠 DataPort 的增量语义（自上次
    拉取以来不重不漏），调度器再按 ``ts`` 水位去重一道（双保险）。
    """

    def __init__(
        self,
        port: DataPort,
        *,
        poll_seconds: float = 30.0,
        now_fn: Callable[[], float] = time.time,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self._port = port
        self._poll_seconds = max(1.0, float(poll_seconds)) if poll_seconds is not None else 30.0
        self._now_fn = now_fn
        self._sleep_fn = sleep_fn
        self._watermarks: dict[str, int] = {}  # label → 已分发最大 bar ts

    def poll(self, item: WatchItem, *, now_utc: int | None = None) -> list[Bar]:
        """拉取该条目自水位以来的新收盘 bar（升序、去重、不重不漏）。"""
        now = int(now_utc if now_utc is not None else self._now_fn())
        tf_sec = timeframe_seconds(item.timeframe)
        wm = self._watermarks.get(item.label)
        fresh: list[Bar] = []
        for bar in self._port.stream_closed_bars(item.symbol, item.timeframe):
            if bar.ts + tf_sec > now:
                continue  # 未收盘（防御：端口层泄漏的 forming bar）
            if wm is not None and bar.ts <= wm:
                continue  # 已分发（去重）
            fresh.append(bar)
        fresh.sort(key=lambda b: b.ts)
        if fresh:
            self._watermarks[item.label] = max(wm or 0, fresh[-1].ts)
        return fresh

    def seed_watermark(self, item: WatchItem, *, now_utc: int | None = None) -> int:
        """回填历史建立水位（启动时调用：历史只进引擎缓冲，**不产出信号**）。

        只回填尾部 ``max(warmup+2, roll_window+2)`` 根：历史 ≥ warmup+1 后
        因子归一化全部走 trailing 窗（与更早历史无关，见对拍 harness 的
        分段重放测试），信号不变；避免对全量历史逐根 O(T) 重算（M1 十万根
        会退化成 O(T²)）。避免「启动即把整段历史当新收盘」的风暴式重算
        ——R2 验收要求「无漏拍、无重复拍」：启动后的第一根**新**收盘 bar
        才是第一个信号。
        """
        now = int(now_utc if now_utc is not None else self._now_fn())
        tf_sec = timeframe_seconds(item.timeframe)
        collected: list[Bar] = []
        for bar in self._port.stream_closed_bars(item.symbol, item.timeframe):
            if bar.ts + tf_sec > now:
                continue
            if collected and bar.ts <= collected[-1].ts:
                continue
            collected.append(bar)
        if not collected:
            self._watermarks[item.label] = 0
            return 0
        warmup = int(item.engine.spec.semantics.warmup_bars)
        need = max(warmup + 2, int(item.engine.spec.semantics.roll_window) + 2)
        for bar in collected[-need:]:
            item.engine.on_closed_bar(bar)  # 回填进引擎历史（可交易状态就绪）
        last_ts = collected[-1].ts
        self._watermarks[item.label] = max(self._watermarks.get(item.label, 0), last_ts)
        return last_ts

    def run(
        self,
        items: list[WatchItem],
        on_bar: Callable[[WatchItem, Bar, float, float], None],
        *,
        once: bool = False,
        stop_after_seconds: float | None = None,
        on_cycle: Callable[[], None] | None = None,
    ) -> None:
        """运行回路：seed 水位 → 轮询分发。

        :param on_bar: ``item, bar, position, prev_position``——prev 在喂引擎
            **之前**捕获，转折检测（方向符号变化）依赖它。
        :param once: 轮询一轮即返回（冒烟/测试用）。
        :param stop_after_seconds: 运行上限（冒烟/守护用）；``None`` = 长驻。
        :param on_cycle: 每个轮询周期开始时回调（与是否有新 bar 无关）——
            纸面/实盘的 kill 文件检查挂在这里（≤ 1 个调度周期生效）。
        """
        for item in items:
            self.seed_watermark(item)
        deadline = None if stop_after_seconds is None else time.monotonic() + stop_after_seconds
        while True:
            if on_cycle is not None:
                on_cycle()
            for item in items:
                for bar in self.poll(item):
                    prev = item.engine.last_position
                    position = item.engine.on_closed_bar(bar)
                    on_bar(item, bar, position, prev)
            if once:
                return
            if deadline is not None and time.monotonic() >= deadline:
                return
            self._sleep_fn(self._poll_seconds)


def log_bar_event(
    item: WatchItem,
    bar: Bar,
    position: float,
    prev_position: float,
    notifier: NotifierPort | None = None,
) -> None:
    """默认的 bar 事件处理：日志 + （可选）转折通知。

    转折 = 方向符号变化（多→空 / 空→多 / 与 0 互转）；观望→观望不打扰。
    脱敏口径：只发信号文字，不发任何金额（R5.2）。
    """
    summary = engine_signal_summary(item.engine, position)
    logger.info("收盘信号 %s bar_ts=%s %s", item.label, bar.ts, summary)
    if notifier is not None and _sign(position) != _sign(prev_position):
        notifier.notify(
            title=f"信号转折 {item.label}",
            body=f"{summary}（bar_ts={bar.ts}）",
            level="info",
        )


def _sign(x: float) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def run_watchlist(
    items: list[WatchItem],
    port: DataPort,
    *,
    notifier: NotifierPort | None = None,
    poll_seconds: float = 30.0,
    once: bool = False,
    stop_after_seconds: float | None = None,
    state_path: Path | None = None,
) -> None:
    """组装默认回路：CloseScheduler + log_bar_event（CLI ``watch run`` 的实现体）。"""
    scheduler = CloseScheduler(port, poll_seconds=poll_seconds)
    scheduler.run(
        items,
        lambda item, bar, position, prev: log_bar_event(item, bar, position, prev, notifier),
        once=once,
        stop_after_seconds=stop_after_seconds,
    )
