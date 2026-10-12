# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for full license text.

"""runtime 包：策略的"装卸码头"（路线图 2.0 R 系列）。

与 ``cli``/``webui`` 同级、允许 IO 的运行时层——把 StrategySpec 变成能在
收盘 bar 上持续出信号、能下单、能对账、能熔断的活系统。研究语义只经
``ir.schema`` + ``core.vm``/``core.vocab``/``core.signal``/``core.features``
公开接口消费，``core/`` 一行不改（G6）。

模块地图::

    ports.py          数据 / 执行 / 通知 三个可插拔抽象
    signal_engine.py  启动安全链 + SignalEngine（收盘 bar → 仓位）
    scheduler.py      收盘调度、去重、watchlist 运行回路
    watchlist.py      挂载清单持久化（R2.3）
    state.py          账本 + 单实例锁 + kill 文件（R3.3）
    risk.py           波动率目标仓位（R3.1）
    paper.py          纸面运行回路（R3）
    reconcile.py      对账器（R3.5）
    sources/          DataPort 实现（MT5 直连）
    notifiers/        NotifierPort 实现（log 兜底；R5 feishu）
    brokers/          ExecutionPort 实现（PaperBroker；R4 MT5Broker 追加）
"""

from __future__ import annotations

from typing import Any

from .ports import Bar, DataPort, ExecutionPort, NotifierPort
from .reconcile import ReconcileReport, reconcile
from .risk import RiskPolicy, VolTargetPolicy, vol_target_lots
from .scheduler import CloseScheduler, WatchItem, run_watchlist, timeframe_seconds
from .signal_engine import SignalEngine, engine_signal_summary, load_spec_checked
from .state import Ledger, LockHandle, kill_file_exists, kill_file_path, state_dir
from .watchlist import WatchEntry, WatchlistStore

__all__ = [
    "Bar",
    "CloseScheduler",
    "DataPort",
    "ExecutionPort",
    "Ledger",
    "LockHandle",
    "MT5DataPort",
    "NotifierPort",
    "PaperBroker",
    "PaperRunner",
    "ReconcileReport",
    "RiskPolicy",
    "SignalEngine",
    "VolTargetPolicy",
    "WatchEntry",
    "WatchItem",
    "WatchlistStore",
    "engine_signal_summary",
    "kill_file_exists",
    "kill_file_path",
    "load_spec_checked",
    "reconcile",
    "run_watchlist",
    "state_dir",
    "timeframe_seconds",
    "vol_target_lots",
]


def __getattr__(name: str) -> Any:  # pragma: no cover - 惰性重依赖
    """惰性导出重依赖实现（避免 import 期触发 MetaTrader5 / 画像链路）。"""
    if name == "MT5DataPort":
        from .sources.mt5 import MT5DataPort

        return MT5DataPort
    if name == "PaperBroker":
        from .brokers.paper import PaperBroker

        return PaperBroker
    if name == "PaperRunner":
        from .paper import PaperRunner

        return PaperRunner
    raise AttributeError(name)
