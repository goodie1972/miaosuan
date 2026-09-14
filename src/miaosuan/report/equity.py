# -*- coding: utf-8 -*-
# Copyright (c) 2026 Original Author
# Copyright (c) 2026 Your Company
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""全样本回测产出层（逐 bar 序列 + 资金曲线 + 滚动夏普 + 交易级统计）。

为什么单开一个模块
------------------
``core/backtest.py`` 的 :class:`MT5Backtest` 已经算出了逐 bar 净收益
``pnl = position * target_ret - turnover * cost_rate``
（``evaluate_fold`` / ``evaluate`` 两处都有），但它**只把 pnl 喂给标量评分
函数就丢掉了** —— 因为搜索只需要一个分数，不需要曲线。

回测页要的是序列，所以这里**新增**一层产出层：复用 ``core`` 的既有语义
（仓位映射、前瞻收益、成本模型、指标函数），只把"被丢弃的 pnl"接出来，
**不修改 ``core`` 的任何已有返回值**。成本计费与画像的
:meth:`~miaosuan.market.profiles.FrozenMarketProfile.apply_cost` 同口径
（**买卖分边**），非对称画像（A 股卖出印花税）下比 core 的单边标量更准。

口径警告（务必遵守）
--------------------
:attr:`~miaosuan.ir.schema.Evidence.val_score` **不是绩效指标**，它是
``ga.py`` 里的复合搜索适应度（年化 / Sortino / Calmar / IC 加权，再乘 OOS
倍率、经 IC 门控）。本模块**从不读它**。真实 Sharpe 由
:func:`miaosuan.report.metrics.compute_metrics` 从逐 bar 净收益算出。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.features import compute_features
from ..core.signal import MIN_TRADE_EXPOSURE, compute_target_positions_stateless
from ..core.vm import StackVM
from ..data.panel import Panel
from ..errors import MiaoSuanError
from ..market.profiles import FrozenMarketProfile
from ..search.mine import compute_target_ret
from .metrics import PerformanceMetrics, compute_metrics

__all__ = [
    "DEFAULT_ROLLING_WINDOW_DIVISOR",
    "BacktestRun",
    "Trade",
    "TradeStats",
    "compute_drawdown",
    "compute_equity_curve",
    "compute_rolling_sharpe",
    "extract_trades",
    "run_full_backtest",
    "to_payload",
]

#: 滚动夏普默认窗口：一年的 1/12（≈1 个月）。H1 外汇下约 520 根 bar。
DEFAULT_ROLLING_WINDOW_DIVISOR = 12

#: 分块 sliding_window_view 时单块允许的最大元素数（控内存峰值，约 32MB）。
_SHARPE_BLOCK_ELEMENTS: int = 4_000_000


@dataclass(frozen=True)
class Trade:
    """一次交易 = **方向翻转后的一段同向持仓**。

    Attributes:
        start: 起始 bar 下标（含）。
        end: 结束 bar 下标（含）。
        direction: ``+1`` 多 / ``-1`` 空。
        pnl: 该段净收益（**已扣成本**，为段内逐 bar pnl 之和）。
        bars: 持仓 bar 数。
    """

    start: int
    end: int
    direction: int
    pnl: float
    bars: int


@dataclass(frozen=True)
class TradeStats:
    """交易级统计。

    Attributes:
        n_trades: 交易笔数（同向持仓段数；方向翻转即新的一笔）。
        n_wins: 盈利笔数。
        win_rate: 胜率 = 盈利笔数 / 总笔数（无交易时为 ``0.0``）。
        avg_win: 平均盈利（盈利段 pnl 的均值；无盈利段为 ``0.0``）。
        avg_loss: 平均亏损（亏损段 pnl 绝对值的均值；无亏损段为 ``0.0``）。
        profit_factor: 盈亏比 = 均盈 / 均亏（**均值的比**，不是总盈/总亏；
            无亏损段时为 ``inf``）。
        best_trade: 最大单笔盈利。
        worst_trade: 最大单笔亏损（负数）。
    """

    n_trades: int = 0
    n_wins: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """序列化为 JSON 友好字典（``inf`` 转成 ``None``，保证是合法 JSON）。"""
        pf = self.profit_factor
        return {
            "n_trades": self.n_trades,
            "n_wins": self.n_wins,
            "win_rate": self.win_rate,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "profit_factor": None if math.isinf(pf) else pf,
            "best_trade": self.best_trade,
            "worst_trade": self.worst_trade,
        }


@dataclass(frozen=True)
class BacktestRun:
    """一次全样本回测的完整结果（逐 bar 序列 + 汇总指标）。

    Attributes:
        symbol: 标的代码（多品种时为首个）。
        symbols: 全部品种名。
        timeframe: 周期标识。
        n_bars: 参与回测的 bar 数。
        cost_rate: 使用的**买入单边**成本率（来自 :class:`CostModel`，非硬编码）。
        periods_per_year: 年化因子（来自 :class:`FrozenMarketProfile`）。
        rolling_window: 滚动夏普窗口（bar 数）。
        times: ``[T]`` bar 时间戳。
        portfolio_pnl: ``[T]`` 组合级逐 bar 净收益（多品种时等权）。
        equity: ``[T]`` 资金曲线 = ``1 + cumsum(portfolio_pnl)``。
        drawdown: ``[T]`` 回撤（正值，``peak - equity``）。
        rolling_sharpe: ``[T]`` 滚动夏普；warmup 段（窗口未满）为 ``nan``。
        metrics: 组合级绩效指标（Sharpe/Sortino/Calmar/MDD/年化/换手）。
        trades: 交易级统计。
        trade_detail: 逐笔明细。
        total_return: 累计收益 = ``cumsum(portfolio_pnl)`` 末值。
        cost_rate_sell: 卖出单边成本率（非对称画像下与 ``cost_rate`` 不同）。
    """

    symbol: str
    symbols: tuple[str, ...]
    timeframe: str
    n_bars: int
    cost_rate: float
    periods_per_year: int
    rolling_window: int
    times: np.ndarray
    portfolio_pnl: np.ndarray
    equity: np.ndarray
    drawdown: np.ndarray
    rolling_sharpe: np.ndarray
    metrics: PerformanceMetrics
    trades: TradeStats
    trade_detail: tuple[Trade, ...] = field(default_factory=tuple)
    total_return: float = 0.0
    #: 卖出单边成本率（非对称画像下与 ``cost_rate`` 不同；对称画像下相等）。
    cost_rate_sell: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """序列化为 JSON 友好字典（序列降精度、``nan`` 转 ``None``）。"""
        return to_payload(self)


def compute_equity_curve(portfolio_pnl: np.ndarray) -> np.ndarray:
    """资金曲线 ``1 + cumsum(pnl)``。

    Args:
        portfolio_pnl: ``[T]`` 逐 bar 净收益。

    Returns:
        ``[T]`` 资金曲线（起点为 1）。
    """
    arr = np.asarray(portfolio_pnl, dtype=np.float64)
    return 1.0 + np.cumsum(arr) if arr.size else np.zeros(0, dtype=np.float64)


def compute_drawdown(equity: np.ndarray) -> np.ndarray:
    """逐点回撤（正值）：``running_peak - equity``。

    Args:
        equity: ``[T]`` 资金曲线。

    Returns:
        ``[T]`` 回撤（非负）。
    """
    arr = np.asarray(equity, dtype=np.float64)
    if arr.size == 0:
        return np.zeros(0, dtype=np.float64)
    peak = np.maximum.accumulate(arr)
    return peak - arr


def compute_rolling_sharpe(
    portfolio_pnl: np.ndarray,
    *,
    window: int,
    periods_per_year: int,
) -> np.ndarray:
    """滚动夏普（**逐窗**方差，ddof=1；分块 ``sliding_window_view`` 控内存）。

    为什么不再用"前缀和 + ``E[x²] − E[x]²``"
    --------------------------------------
    旧实现先对**整段**去全局均值，再用 ``E[x²] − E[x]²`` 求窗内方差。当窗口正好落在
    一段常数区间上时，两个 O(1) 量相减发生**灾难性抵消**：残余 std ≈ 3e-10 远大于真值
    0，Sharpe 被放大到 ~2e9（实测拼接常数段 ``[0.005×1000, 0.001×1000]``、``window=520``
    时 max|S| ≈ 2.3e9，峰值恰在 ptp=0 的常数窗口上）。

    改为**在每个窗口内部**各自求均值与 ``var(ddof=1)``：常数窗口得到精确 0，不再爆炸。
    ``ddof`` 取 1 与 :func:`miaosuan.report.metrics.sharpe` 对齐（口径一致）。

    Args:
        portfolio_pnl: ``[T]`` 逐 bar 净收益。
        window: 滑窗长度（bar 数）。
        periods_per_year: 年化因子。

    Returns:
        ``[T]`` 滚动夏普；前 ``window - 1`` 个位置为 ``nan``（窗口未满），
        窗口内标准差过小（``< 1e-12``）时该点记 ``0.0``（避免除零产生 inf）。
    """
    arr = np.asarray(portfolio_pnl, dtype=np.float64)
    n = arr.size
    out = np.full(n, np.nan, dtype=np.float64)
    if n < window or window < 2:
        return out
    # sliding_window_view 是**视图**（不复制）；对每个窗口独立求 mean / std(ddof=1)。
    slide = np.lib.stride_tricks.sliding_window_view(arr, window)
    n_win = slide.shape[0]
    sharpe = np.empty(n_win, dtype=np.float64)
    scale = math.sqrt(float(periods_per_year))
    # 分块处理：避免一次性物化 (n_win, window) 的临时大数组。
    block = max(1, _SHARPE_BLOCK_ELEMENTS // window)
    for start in range(0, n_win, block):
        stop = min(start + block, n_win)
        chunk = slide[start:stop]
        mean = chunk.mean(axis=1)
        std = chunk.std(axis=1, ddof=1)
        active = std > 1e-12
        sharpe[start:stop] = np.where(
            active, mean / np.where(active, std, 1.0) * scale, 0.0
        )
    out[window - 1 :] = sharpe
    return out


def extract_trades(
    position: np.ndarray,
    pnl: np.ndarray,
    *,
    threshold: float = MIN_TRADE_EXPOSURE,
) -> tuple[Trade, ...]:
    """把连续仓位切成"一次交易 = 一段同向持仓"，并累加段内净收益。

    方向判定沿用 :func:`~miaosuan.core.signal.target_to_direction` 的阈值语义：
    ``|position| < threshold`` 视为空仓（0），否则取符号。

    区间划分（**无重叠、无遗漏**，保证 ``sum(t.pnl) == pnl.sum()``）
    --------------------------------------------------------------
    旧的实现只在"方向相同"的 bar 上累计 pnl，于是**归零那一根 bar** 的换手成本
    既不属于旧单也不属于新单，直接掉进缝里：实测 ``pos=+1×6→0×3→−1×6``、
    ``cost=3e-4`` 时 ``sum(trades)=−6e-4`` 而 ``total=−9e-4``，缺口正是归零 bar 的
    成本。修正后的边界归属规则（与 lead 口径一致）：

    * **平仓转空仓**：归零那一根 bar 属于**旧单**（其换手成本计入旧单，旧单右端含该 bar）；
    * **直接翻转**（``+1 → −1``）：翻转那一根 bar 属于**新单**；
    * 两笔之间方向为空的 bar（净收益恒为 0）并入**后一笔**，使区间首尾相接、无缝隙。

    Args:
        position: ``[T]`` 连续目标仓位。
        pnl: ``[T]`` 逐 bar 净收益（**已扣成本**）。
        threshold: 中性带阈值（低于此值视为空仓）。

    Returns:
        逐笔 :class:`Trade`（按时间升序、区间首尾相接）。
    """
    pos = np.asarray(position, dtype=np.float64).reshape(-1)
    net = np.asarray(pnl, dtype=np.float64).reshape(-1)
    t = min(pos.size, net.size)
    if t == 0:
        return ()

    direction = np.zeros(t, dtype=np.int8)
    direction[pos[:t] >= threshold] = 1
    direction[pos[:t] <= -threshold] = -1

    trades: list[Trade] = []
    prev_end = -1
    i = 0
    while i < t:
        d = int(direction[i])
        if d == 0:
            i += 1
            continue
        j = i + 1
        while j < t and int(direction[j]) == d:
            j += 1
        # j 是第一个与 d 不同的 bar（或 t）。若它是"转空仓"的归零 bar，则并入本笔
        # （平仓成本）；若是直接翻转或序列结束，则本笔止于持仓段最后一根 bar，
        # 翻转 bar 留给下一笔。
        end = j if (j < t and int(direction[j]) == 0) else j - 1
        # 首笔从首个持仓 bar 起；其后每笔从上一笔右端 +1 起，把两笔之间的空仓 bar
        # （净收益恒为 0）并入后一笔 —— 区间连续、无重叠、无缝隙。
        start = i if prev_end < 0 else prev_end + 1
        trades.append(
            Trade(
                start=start,
                end=end,
                direction=d,
                pnl=float(net[start : end + 1].sum()),
                bars=int(end - start + 1),
            )
        )
        prev_end = end
        i = end + 1
    return tuple(trades)


def summarize_trades(trades: tuple[Trade, ...]) -> TradeStats:
    """由逐笔明细汇总 :class:`TradeStats`。

    Args:
        trades: :func:`extract_trades` 的输出。

    Returns:
        交易级统计；无交易时全为零值。
    """
    if not trades:
        return TradeStats()
    pnls = [tr.pnl for tr in trades]
    wins = [p for p in pnls if p > 0.0]
    losses = [-p for p in pnls if p < 0.0]
    avg_win = float(sum(wins) / len(wins)) if wins else 0.0
    avg_loss = float(sum(losses) / len(losses)) if losses else 0.0
    profit_factor = math.inf if (wins and not losses) else (
        avg_win / avg_loss if avg_loss > 0.0 else 0.0
    )
    return TradeStats(
        n_trades=len(trades),
        n_wins=len(wins),
        win_rate=float(len(wins) / len(trades)),
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=float(profit_factor),
        best_trade=float(max(pnls)),
        worst_trade=float(min(pnls)),
    )


def run_full_backtest(
    tokens: Any,
    panel: Panel,
    profile: FrozenMarketProfile,
    *,
    rolling_window: int | None = None,
    neutral_band: float = MIN_TRADE_EXPOSURE,
    long_only: bool | None = None,
) -> BacktestRun:
    """在**全样本**上跑一次回测，返回逐 bar 序列与汇总指标。

    计算链（全部复用 ``core`` 既有语义，不改任何已有返回值）::

        features  = compute_features(panel.to_raw_dict())          # [N, F, T]
        factor    = StackVM().execute(tokens, features)            # [N, T]
        position  = tanh(factor)  （过中性带置 0）                  # [N, T]
        ret       = log(open[t+2] / open[t+1])                     # [N, T]
        pnl       = profile.apply_cost(position, ret)              # [N, T]
                    = position*ret - (buy_rate·Δpos⁺ + sell_rate·Δpos⁻)

    Args:
        tokens: 因子 token 序列（来自 ``StrategySpec.payload.tokens``）。
        panel: 行情面板。
        profile: 市场画像（提供成本率、年化 bar 数、是否只做多）。
        rolling_window: 滚动夏普窗口；``None`` 时用 ``bars_per_year // 12``。
        neutral_band: 中性带阈值（低于此值不持仓）。
        long_only: 是否只做多；``None`` 时取 ``profile.long_only``。

    Returns:
        :class:`BacktestRun`。

    Raises:
        MiaoSuanError: 公式不可求值（StackVM 返回 ``None``）。
    """
    raw = panel.to_raw_dict()
    features = compute_features(raw)
    factor = StackVM().execute([int(t) for t in tokens], features)
    if factor is None:
        raise MiaoSuanError(f"公式不可求值（tokens={list(tokens)!r}）")

    factor = np.asarray(factor, dtype=np.float64)
    ret = np.asarray(compute_target_ret(panel.open), dtype=np.float64)
    # 前瞻收益末 2 列无未来数据（置 0），对齐到共同长度避免尾部失真。
    n_bars = int(min(factor.shape[1], ret.shape[1]))
    factor = factor[:, :n_bars]
    ret = ret[:, :n_bars]

    position = compute_target_positions_stateless(
        factor,
        min_trade_exposure=float(neutral_band),
        long_only=bool(profile.long_only if long_only is None else long_only),
    )

    # 成本按**买卖分边**计费，直接复用画像的 apply_cost（与 core 的 ``Δpos⁺/Δpos⁻``
    # 分边口径一致）。非对称画像（A 股卖出印花税）下，若用单边 buy_rate×总换手，
    # 卖出腿会被按买入价计费 —— 曲线与门禁就会对不上。
    pnl = np.asarray(profile.apply_cost(position, ret), dtype=np.float64)
    # 报告口径：记录买入单边率（对称画像下即单边成本率）；卖出率见 ``to_payload``。
    cost_rate = float(profile.cost_model.buy_rate())

    periods_per_year = int(profile.bars_per_year)
    window = (
        int(rolling_window)
        if rolling_window
        else max(2, periods_per_year // DEFAULT_ROLLING_WINDOW_DIVISOR)
    )

    # 组合级：多品种等权（N=1 时等于该品种自身）。
    portfolio_pnl = pnl.mean(axis=0) if pnl.shape[0] else np.zeros(n_bars)
    portfolio_pos = position.mean(axis=0) if position.shape[0] else np.zeros(n_bars)

    equity = compute_equity_curve(portfolio_pnl)
    drawdown = compute_drawdown(equity)
    rolling = compute_rolling_sharpe(
        portfolio_pnl, window=window, periods_per_year=periods_per_year
    )
    metrics = compute_metrics(
        portfolio_pnl, periods_per_year=periods_per_year, position=portfolio_pos
    )
    trades = extract_trades(portfolio_pos, portfolio_pnl, threshold=float(neutral_band))
    times = np.asarray(panel.time)[:n_bars]

    return BacktestRun(
        symbol=panel.symbols[0] if panel.symbols else "",
        symbols=tuple(panel.symbols),
        timeframe=panel.timeframe,
        n_bars=n_bars,
        cost_rate=cost_rate,
        periods_per_year=periods_per_year,
        rolling_window=window,
        times=times,
        portfolio_pnl=portfolio_pnl,
        equity=equity,
        drawdown=drawdown,
        rolling_sharpe=rolling,
        metrics=metrics,
        trades=summarize_trades(trades),
        trade_detail=trades,
        total_return=float(portfolio_pnl.sum()),
        cost_rate_sell=float(profile.cost_model.sell_rate()),
    )


def _peak(values: np.ndarray) -> float:
    """序列峰值（忽略非有限值）；空序列返回 ``0.0``。"""
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = arr[np.isfinite(arr)]
    return float(finite.max()) if finite.size else 0.0


def _series(values: np.ndarray, digits: int = 6) -> list[Any]:
    """把序列转成 JSON 友好列表（``nan`` / ``inf`` → ``None``）。"""
    out: list[Any] = []
    for value in np.asarray(values, dtype=np.float64).reshape(-1):
        if not math.isfinite(float(value)):
            out.append(None)
        else:
            out.append(round(float(value), digits))
    return out


def to_payload(result: BacktestRun, *, max_trades: int = 200) -> dict[str, Any]:
    """把 :class:`BacktestRun` 序列化为可写盘 / 可回传的 dict。

    Args:
        result: 回测结果。
        max_trades: 明细最多保留多少笔（防止大样本把 JSON 撑爆）。

    Returns:
        JSON 安全字典（无 ``nan`` / ``inf``）。
    """
    detail = result.trade_detail[:max_trades]
    return {
        "meta": {
            "symbol": result.symbol,
            "symbols": list(result.symbols),
            "timeframe": result.timeframe,
            "n_bars": result.n_bars,
            "cost_rate": result.cost_rate,
            "cost_rate_sell": result.cost_rate_sell,
            "periods_per_year": result.periods_per_year,
            "rolling_window": result.rolling_window,
            # 口径声明：页面必须照此标注，禁止把 val_score 当绩效展示。
            "caliber": "逐 bar 净收益 = position*ret - (buy_rate·Δpos⁺ + sell_rate·Δpos⁻)"
                       "（买卖分边计费，同 profile.apply_cost）；"
                       "Sharpe/Sortino 由该序列算出，与 val_score（搜索适应度）无关",
        },
        "summary": {
            "total_return": result.total_return,
            **result.metrics.to_dict(),
            # metrics.max_drawdown 是**相对**回撤 (peak-equity)/peak；这里是
            # 序列口径的**绝对**回撤峰值（权益单位）。两者单位不同，页面上
            # 必须分开标注，否则会出现"卡片 0.148 / 图上 0.206"的自相矛盾。
            "max_drawdown_abs": _peak(result.drawdown),
            **result.trades.to_dict(),
        },
        "series": {
            "time": [int(t) for t in np.asarray(result.times).reshape(-1)],
            "equity": _series(result.equity),
            "drawdown": _series(result.drawdown),
            "rolling_sharpe": _series(result.rolling_sharpe),
        },
        "trades": [
            {
                "start": tr.start,
                "end": tr.end,
                "direction": tr.direction,
                "pnl": round(tr.pnl, 6),
                "bars": tr.bars,
            }
            for tr in detail
        ],
        "trades_truncated": len(result.trade_detail) > len(detail),
    }
