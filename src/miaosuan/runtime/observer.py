# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for full license text.

"""本地 runtime 观测（路线图 R5.1）：webui 的 ``local`` 数据通道。

职责：从 **本地真实状态** 读取观测数据，替代 R5 之前的「示值」假数据桩：

* 引擎状态 → watchlist 条目 + 门禁 verdict + 纸面账本持仓 + kill 文件；
* 报价     → 本地缓存最后一根已收盘 bar 的收盘价（无 MT5 tick 时的诚实口径，
  显式标注 ``source="cached_last_close"``，**绝不伪造 0 报价**）；
* K 线     → 本地缓存 parquet/csv 的真实 K 线；
* 信号     → 对每个 watch 条目回放 :class:`SignalEngine`，产出最新仓位/方向。

可靠性约定：本模块 **永不 500**——任何读取失败折叠为 :class:`MiaoSuanError`，
由 webui 包装成 ``{ok: False, reason, error}`` 信封；找不到数据与「数据为零」
严格区分（后者如实返回，前者报 ``no_local_data``）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import artifacts_dir, kline_data_dir
from ..data.loader import infer_symbol_timeframe, load
from ..errors import MiaoSuanError
from .signal_engine import SignalEngine
from .state import kill_file_exists, state_dir
from .watchlist import WatchlistStore

__all__ = [
    "local_engine_status",
    "local_market_candles",
    "local_market_price",
    "local_signals",
    "cached_kline_files",
]

_DATA_SUFFIXES = {".parquet", ".pq", ".csv"}
_KLINE_DIRS: tuple[Path, ...] = (Path(kline_data_dir()), Path(artifacts_dir()).parent / "data")
#: 报价格式化位数（XAUUSD 等贵金属两位小数；未知按原值回传）。
_DIGITS = 2


def cached_kline_files(symbol: str, timeframe: str) -> list[Path]:
    """在本地行情目录中按文件名契约 ``{symbol}_{timeframe}`` 找匹配文件。

    与 :func:`miaosuan.data.loader.infer_symbol_timeframe` 同口径（备注段忽略）。
    找不到任何匹配返回空列表（调用方决定如何降级），**不抛**。
    """
    wanted_tf = (timeframe or "").upper()
    hits: list[Path] = []
    for directory in _KLINE_DIRS:
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            if path.suffix.lower() not in _DATA_SUFFIXES:
                continue
            file_symbol, file_tf = infer_symbol_timeframe(path)
            if file_symbol.upper() != (symbol or "").upper():
                continue
            if wanted_tf and file_tf != wanted_tf:
                continue
            hits.append(path)
    return hits


def _load_candles(symbol: str, timeframe: str) -> list[dict[str, Any]]:
    """读取本地缓存 K 线为 ``{time, open, high, low, close, volume}`` 列表。

    :raises MiaoSuanError: 无匹配文件 / 文件读失败。
    """
    files = cached_kline_files(symbol, timeframe)
    if not files:
        raise MiaoSuanError(
            f"本地无 {symbol}/{timeframe} 的缓存 K 线",
            context={"symbol": symbol, "timeframe": timeframe},
        )
    last_error: MiaoSuanError | None = None
    for path in files:
        try:
            panel = load(path)
        except Exception as exc:  # noqa: BLE001 - 逐文件尝试，全部失败才报错
            last_error = MiaoSuanError(
                f"本地 K 线读取失败: {path.name}（{exc}）",
                context={"path": str(path)},
            )
            continue
        n = panel.n_bars
        if n == 0:
            continue
        out: list[dict[str, Any]] = []
        for i in range(n):
            out.append(
                {
                    "time": int(panel.time[i]),
                    "open": round(float(panel.open[0, i]), _DIGITS),
                    "high": round(float(panel.high[0, i]), _DIGITS),
                    "low": round(float(panel.low[0, i]), _DIGITS),
                    "close": round(float(panel.close[0, i]), _DIGITS),
                    "volume": round(float(panel.volume[0, i]), 4),
                }
            )
        return out
    if last_error is not None:
        raise last_error
    raise MiaoSuanError(
        f"本地缓存为空: {symbol}/{timeframe}",
        context={"symbol": symbol, "timeframe": timeframe},
    )


def local_market_candles(symbol: str, timeframe: str, count: int) -> list[dict[str, Any]]:
    """返回**最新** ``count`` 根已收盘 K 线（升序、只读）。

    :raises MiaoSuanError: 无本地数据 / 读失败（webui 包装成降级信封）。
    """
    candles = _load_candles(symbol, timeframe)
    bounded = max(1, min(int(count), len(candles)))
    return candles[-bounded:]


def local_market_price(symbol: str) -> dict[str, Any]:
    """本地报价：取缓存最后一根收盘价，显式标注来源。

    无 MT5 tick 连接时这是**最诚实**的本地报价——bid/ask 均取最新收盘价，
    ``spread`` 恒为 0 且 ``source="cached_last_close"`` 明示「非实时 tick」。

    :raises MiaoSuanError: 无本地数据（webui 包装成降级信封，绝不伪造 0）。
    """
    candles = _load_candles(symbol, "")
    last = candles[-1]["close"]
    return {
        "symbol": symbol,
        "bid": last,
        "ask": last,
        "spread": 0.0,
        "source": "cached_last_close",
        "ts": candles[-1]["time"],
    }


def _spec_gate_verdict(spec_path: Path) -> str:
    """读 spec 的 gate_verdict（文件缺失/坏 JSON → ``"unavailable"``，不抛）。"""
    try:
        from ..ir.codec import read_spec

        return read_spec(spec_path).evidence.gate_verdict
    except Exception:  # noqa: BLE001 - 观测路径绝不因单条 spec 失败
        return "unavailable"


def local_engine_status() -> dict[str, Any]:
    """本地引擎状态：watchlist 条目 + 纸面账本持仓 + kill 文件（全只读）。

    永不抛：任何单项读取失败降级为该字段的 ``unavailable`` 值。
    """
    entries = WatchlistStore().load()
    watch_items: list[dict[str, Any]] = []
    for e in entries:
        spec_path = Path(e.spec_path)
        watch_items.append(
            {
                "symbol": e.symbol,
                "timeframe": e.timeframe,
                "profile": e.profile,
                "spec_path": e.spec_path,
                "spec_exists": spec_path.is_file(),
                "gate_verdict": _spec_gate_verdict(spec_path) if spec_path.is_file() else "unavailable",
            }
        )

    # 纸面账本持仓（state_dir 下 paper_*.json；与 paper.py 的命名契约一致）。
    positions: list[dict[str, Any]] = []
    from .paper import paper_ledger_path  # 延迟导入避免重依赖

    st_dir = state_dir()
    if st_dir.is_dir():
        from .state import Ledger

        for ledger_file in sorted(st_dir.glob("paper_*.json")):
            try:
                ledger = Ledger(ledger_file)
                positions.extend(ledger.open_positions())
                ledger.total_fees()  # 触发加载（不存在时返回空）
            except Exception:  # noqa: BLE001 - 观测路径绝不因单本账本失败
                continue

    kill = kill_file_exists()
    return {
        "status": "local_runtime",
        "bridge_connected": True,  # 本地通道无桥接概念：恒真（banner 语义用）
        "watchlist": watch_items,
        "positions": positions,
        "kill_file": kill,
        "kill_pending": kill,
        "started_at": "",
        "uptime_seconds": 0,
    }


def local_signals() -> list[dict[str, Any]]:
    """对每个 watch 条目：回放本地缓存 K 线 → 最新目标仓位/方向。

    返回形状对齐 webui 前端 ``renderRtSignals``（``timestamp/strategy/magic/
    signal/score_long/score_short/status/ticket/price_entry``）。

    单条失败不影响其他条目（该项 ``status="unavailable"`` 并附原因）。
    """
    entries = WatchlistStore().load()
    out: list[dict[str, Any]] = []
    for e in entries:
        spec_path = Path(e.spec_path)
        label = f"{e.symbol}.{e.timeframe}"
        try:
            from ..ir.codec import read_spec

            spec = read_spec(spec_path)
        except Exception as exc:  # noqa: BLE001
            out.append(_signal_row(label, status="unavailable", reason=f"spec: {exc}"))
            continue
        if spec.evidence.gate_verdict != "DEPLOYABLE":
            out.append(
                _signal_row(
                    label,
                    status="blocked",
                    reason=f"gate={spec.evidence.gate_verdict}",
                )
            )
            continue
        try:
            candles = _load_candles(e.symbol, e.timeframe or spec.semantics.timeframe)
            engine = SignalEngine(spec)
            last_bar_ts = candles[-1]["time"]
            # 回放全部缓存 bar（观测路径：量小逐根消费，量大致多；引擎内部全历史重算）
            position = 0.0
            for bar in candles:
                position = engine.on_closed_bar(
                    _bar_from_candle(e.symbol, e.timeframe or spec.semantics.timeframe, bar)
                )
            direction = "多" if position > 0 else ("空" if position < 0 else "观望")
            magic = int(spec.spec_id[:8], 16) % 2_147_483_646 + 1
            out.append(
                {
                    "timestamp": last_bar_ts,
                    "strategy": spec.name,
                    "magic": magic,
                    "signal": direction,
                    "score_long": round(max(position, 0.0), 4),
                    "score_short": round(max(-position, 0.0), 4),
                    "status": "signal",
                    "ticket": "",
                    "price_entry": "",
                    "position": round(position, 4),
                    "symbol": e.symbol,
                    "timeframe": e.timeframe or spec.semantics.timeframe,
                }
            )
        except MiaoSuanError as exc:
            out.append(_signal_row(label, status="no_data", reason=str(exc)))
        except Exception as exc:  # noqa: BLE001 - 观测路径逐条容错
            out.append(_signal_row(label, status="unavailable", reason=str(exc)))
    return out


def _signal_row(label: str, *, status: str, reason: str = "") -> dict[str, Any]:
    """降级信号行（不携带任何假仓位）。"""
    symbol, _, tf = label.partition(".")
    return {
        "timestamp": 0,
        "strategy": label,
        "magic": "",
        "signal": "—",
        "score_long": 0.0,
        "score_short": 0.0,
        "status": status,
        "reason": reason,
        "ticket": "",
        "price_entry": "",
        "symbol": symbol,
        "timeframe": tf,
    }


def _bar_from_candle(symbol: str, timeframe: str, candle: dict[str, Any]) -> Any:
    """把本地缓存蜡烛转成 runtime 的 :class:`Bar`（收盘纪律：缓存即已收盘）。"""
    from .ports import Bar

    return Bar(
        symbol=symbol,
        timeframe=timeframe,
        ts=int(candle["time"]),
        open=float(candle["open"]),
        high=float(candle["high"]),
        low=float(candle["low"]),
        close=float(candle["close"]),
        volume=float(candle["volume"]),
        closed=True,
    )