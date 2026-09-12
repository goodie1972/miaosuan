"""回测产出层（``report/equity.py``）—— 序列、资金曲线、滚动夏普、交易统计。

回归防线的重点：

* ``pnl`` 公式必须与 ``core/backtest.py`` **逐字一致**
  （``position * ret - turnover * cost_rate``），否则"回测页的曲线"和
  "搜索时用的评分"就是两套数，页面上再漂亮也没有意义；
* 成本率必须来自画像的 ``CostModel``，不许硬编码；
* 序列化后**不能出现 NaN/Infinity** —— 那不是合法 JSON，前端会整页白屏。
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from miaosuan.core.features import compute_features
from miaosuan.core.signal import MIN_TRADE_EXPOSURE, compute_target_positions_stateless
from miaosuan.core.vm import StackVM
from miaosuan.data.panel import Panel
from miaosuan.errors import MiaoSuanError
from miaosuan.market.profiles import FOREX_XAUUSD
from miaosuan.report.equity import (
    compute_drawdown,
    compute_equity_curve,
    compute_rolling_sharpe,
    extract_trades,
    run_full_backtest,
    summarize_trades,
    to_payload,
)
from miaosuan.report.metrics import sharpe
from miaosuan.search.mine import compute_target_ret

TOKENS: tuple[int, ...] = (33, 62, 3, 87, 72, 119, 73, 103)


def _panel(n_bars: int = 600, *, seed: int = 7) -> Panel:
    """造一个确定性面板（N=1），价格随机游走。"""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, 0.002, size=n_bars)
    close = 2000.0 * np.exp(np.cumsum(steps))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) * 1.001
    low = np.minimum(open_, close) * 0.999
    volume = np.full(n_bars, 1000.0)
    return Panel(
        open=open_.reshape(1, -1).astype(np.float32),
        high=high.reshape(1, -1).astype(np.float32),
        low=low.reshape(1, -1).astype(np.float32),
        close=close.reshape(1, -1).astype(np.float32),
        volume=volume.reshape(1, -1).astype(np.float32),
        time=np.arange(n_bars, dtype=np.int64) * 3600 + 1_600_000_000,
        symbols=("XAUUSD",),
        timeframe="H1",
        market_profile_name="FOREX_XAUUSD",
    )


def test_pnl_matches_core_backtest_formula() -> None:
    """逐 bar 净收益必须与 ``core/backtest.py`` 的公式**逐字一致**。

    ``core`` 里是 ``position * target_ret - turnover * cost_rate``
    （``evaluate_fold`` / ``evaluate`` 两处）。这里若写成别的（漏掉成本、或用
    gross 收益当净收益），回测页的数字就和搜索时用的不是一套 —— 页面看起来
    依然"正常"，只能靠对齐公式来防。
    """
    panel = _panel()
    result = run_full_backtest(TOKENS, panel, FOREX_XAUUSD)

    factor = np.asarray(
        StackVM().execute(list(TOKENS), compute_features(panel.to_raw_dict())), dtype=np.float64
    )
    ret = np.asarray(compute_target_ret(panel.open), dtype=np.float64)
    n_bars = min(factor.shape[1], ret.shape[1])
    factor = factor[:, :n_bars]
    ret = ret[:, :n_bars]

    position = compute_target_positions_stateless(
        factor, min_trade_exposure=MIN_TRADE_EXPOSURE, long_only=FOREX_XAUUSD.long_only
    )
    prev_pos = np.roll(position, 1, axis=1)
    prev_pos[:, 0] = 0.0
    turnover = np.abs(position - prev_pos)
    expected = (position * ret - turnover * result.cost_rate).mean(axis=0)

    np.testing.assert_allclose(result.portfolio_pnl, expected, rtol=0.0, atol=1e-12)
    # 关键：成本率必须来自画像，不是硬编码（0.0001 佣金 + 0.0002 滑点）
    assert result.cost_rate == pytest.approx(FOREX_XAUUSD.cost_model.buy_rate())
    assert result.cost_rate == pytest.approx(0.0003)


def test_cost_rate_comes_from_profile_not_hardcoded() -> None:
    """成本率必须取自画像 CostModel（换画像 → 成本跟着变）。"""
    panel = _panel(n_bars=200)
    cheap = replace(FOREX_XAUUSD, cost_model=FOREX_XAUUSD.cost_model.scaled(0.0))
    result = run_full_backtest(TOKENS, panel, cheap)
    costly = run_full_backtest(TOKENS, panel, FOREX_XAUUSD)

    assert result.cost_rate == 0.0
    assert costly.cost_rate == pytest.approx(0.0003)
    # turnover ≥ 0，因此零成本的逐 bar 净收益必然 ≥ 有成本时
    assert result.total_return >= costly.total_return


def test_equity_curve_and_drawdown() -> None:
    """资金曲线 = 1 + cumsum；回撤 = peak - equity，非负。"""
    pnl = np.array([0.1, -0.05, 0.02, -0.2], dtype=np.float64)
    equity = compute_equity_curve(pnl)
    assert equity[0] == pytest.approx(1.1)
    assert equity[-1] == pytest.approx(1.0 - 0.13)
    dd = compute_drawdown(equity)
    assert (dd >= 0).all()
    assert dd.max() == pytest.approx(1.1 - 0.87)

    assert compute_equity_curve(np.zeros(0)).size == 0
    assert compute_drawdown(np.zeros(0)).size == 0


def test_rolling_sharpe_warmup_is_nan_and_no_inf() -> None:
    """窗口未满处必须是 nan（前端留空），且窗口内 std=0 时不得产生 inf。"""
    rng = np.random.default_rng(3)
    pnl = rng.normal(0.001, 0.01, size=100)
    out = compute_rolling_sharpe(pnl, window=20, periods_per_year=6240)
    assert out.size == 100
    assert all(math.isnan(v) for v in out[:19])
    assert all(math.isfinite(v) for v in out[19:])

    flat = np.full(50, 0.01)
    flat_out = compute_rolling_sharpe(flat, window=10, periods_per_year=6240)
    # 常数序列 std=0 → 该点记 0.0，而不是 inf
    assert flat_out[9] == 0.0
    assert not any(math.isinf(v) for v in flat_out if not math.isnan(v))

    # 样本短于窗口 → 全 nan（前端走空状态）
    short = compute_rolling_sharpe(np.array([0.1, 0.2]), window=10, periods_per_year=6240)
    assert all(math.isnan(v) for v in short)


def test_rolling_sharpe_constant_segments_do_not_explode() -> None:
    """拼接常数段不得把滚动夏普放大成天文数字（回归 B1）。

    旧实现（全局中心化 + ``E[x²]−E[x]²``）在**常数窗口**上发生灾难性抵消：实测
    该序列 ``window=520``、``ppy=8760`` 时 max|S|≈2.3e9，峰值恰好落在一个 ptp=0
    的常数窗口上。改为逐窗 ``var(ddof=1)`` 后常数窗口精确为 0，整段不再爆炸。
    """
    arr = np.concatenate([np.full(1000, 0.005), np.full(1000, 0.001)])
    window = 520

    out = compute_rolling_sharpe(arr, window=window, periods_per_year=1)
    # 完全落在同一常数段内的窗口 → 精确 0（不是 1e9，也不是 inf/nan）
    for i in range(window - 1, 1000):          # 第一段 0.005 内的满窗
        assert out[i] == 0.0, (i, out[i])
    for i in range(1000 + window - 1, 2000):   # 第二段 0.001 内的满窗
        assert out[i] == 0.0, (i, out[i])
    # 验收：整段 max|S| < 100（ppy=1，隔离年化因子、只测数值稳健性）
    assert float(np.nanmax(np.abs(out))) < 100.0
    # 真实年化因子下也不再是 2.3e9 那种量级（远小于旧的爆炸值）
    out_real = compute_rolling_sharpe(arr, window=window, periods_per_year=8760)
    assert float(np.nanmax(np.abs(out_real))) < 1e4


def test_rolling_sharpe_matches_per_window_reference() -> None:
    """与逐窗 ``np.std(ddof=1)`` 朴素参考实现逐点一致（max|Δ| < 1e-9）。"""
    rng = np.random.default_rng(11)
    arr = np.concatenate(
        [rng.normal(0.0, 0.01, 300), np.full(50, 0.004), rng.normal(0.0, 0.02, 250)]
    )
    window, ppy = 30, 6240
    out = compute_rolling_sharpe(arr, window=window, periods_per_year=ppy)

    ref = np.full(arr.size, np.nan)
    for i in range(window - 1, arr.size):
        w = arr[i - window + 1 : i + 1]
        std = float(w.std(ddof=1))
        ref[i] = 0.0 if std < 1e-12 else float(w.mean() / std * math.sqrt(ppy))

    assert np.array_equal(np.isnan(out), np.isnan(ref))
    finite = ~np.isnan(out)
    assert float(np.max(np.abs(out[finite] - ref[finite]))) < 1e-9


def test_rolling_sharpe_ddof_matches_metrics_sharpe() -> None:
    """任一满窗的滚动值 == ``metrics.sharpe(该窗口)``（ddof=1 口径对齐）。"""
    rng = np.random.default_rng(5)
    arr = rng.normal(0.001, 0.01, size=200)
    window, ppy = 25, 6240
    out = compute_rolling_sharpe(arr, window=window, periods_per_year=ppy)
    i = 137
    assert out[i] == pytest.approx(sharpe(arr[i - window + 1 : i + 1], ppy))


def test_extract_trades_splits_on_direction_flip() -> None:
    """一次交易 = 一段同向持仓；方向翻转 / 回到空仓都开新的一笔。"""
    pos = np.array([0.0, 0.5, 0.5, 0.0, -0.5, -0.5, 0.8])
    pnl = np.array([0.0, 0.1, 0.1, 0.0, -0.2, -0.2, 0.3])
    trades = extract_trades(pos, pnl, threshold=MIN_TRADE_EXPOSURE)
    assert [t.direction for t in trades] == [1, -1, 1]
    assert trades[0].pnl == pytest.approx(0.2)
    assert trades[1].pnl == pytest.approx(-0.4)
    assert trades[0].bars == 2
    assert trades[2].start == 6 and trades[2].end == 6


def test_extract_trades_ignores_sub_threshold_wiggle() -> None:
    """中性带以内的抖动不算持仓，因此不会碎成一堆假交易。"""
    pos = np.array([0.5, 0.04, 0.5, 0.0])
    pnl = np.array([0.1, 0.001, 0.1, 0.0])
    trades = extract_trades(pos, pnl, threshold=MIN_TRADE_EXPOSURE)
    # 0.04 < 0.05（中性带）→ 该 bar 方向归零 → 一笔拆成两笔
    assert len(trades) == 2
    assert all(t.direction == 1 for t in trades)
    assert trades[0].bars == 1 and trades[1].bars == 1


def test_summarize_trades_win_rate_and_profit_factor() -> None:
    """胜率 = 盈利笔数/总笔数；盈亏比 = 均盈/均亏（**均值的比**）。"""
    trades = extract_trades(
        np.array([1.0, 1.0, -1.0, -1.0, 1.0, 1.0]),
        np.array([0.10, 0.10, -0.05, -0.05, 0.02, 0.02]),
        threshold=0.5,
    )
    stats = summarize_trades(trades)
    assert stats.n_trades == 3
    assert stats.n_wins == 2
    assert stats.win_rate == pytest.approx(2 / 3)
    # 两段盈利：0.20 与 0.04 → 均盈 0.12；一段亏损 0.10 → 均亏 0.10
    assert stats.avg_win == pytest.approx(0.12)
    assert stats.avg_loss == pytest.approx(0.10)
    assert stats.profit_factor == pytest.approx(1.2)
    assert stats.best_trade == pytest.approx(0.20)
    assert stats.worst_trade == pytest.approx(-0.10)


def test_summarize_trades_no_losses_is_inf_not_zero() -> None:
    """全盈利时盈亏比是 ``inf``（不是 0），序列化时转 ``None``。"""
    stats = summarize_trades(extract_trades(np.array([1.0, 1.0]), np.array([0.1, 0.1]), threshold=0.5))
    assert math.isinf(stats.profit_factor)
    assert stats.to_dict()["profit_factor"] is None


def test_summarize_trades_empty() -> None:
    """无交易时全零，且 win_rate 不是 NaN。"""
    stats = summarize_trades(())
    assert stats.n_trades == 0
    assert stats.win_rate == 0.0
    assert stats.to_dict()["win_rate"] == 0.0


def test_run_full_backtest_raises_on_unevaluable_formula() -> None:
    """公式不可求值（StackVM 返回 None）必须显式报错，不能静默画一条平线。"""
    with pytest.raises(MiaoSuanError):
        run_full_backtest((999999,), _panel(n_bars=100), FOREX_XAUUSD)


def test_run_full_backtest_series_lengths_aligned() -> None:
    """time / equity / drawdown / rolling_sharpe 四条序列长度必须一致。"""
    result = run_full_backtest(TOKENS, _panel(n_bars=400), FOREX_XAUUSD, rolling_window=50)
    assert result.n_bars == 400
    for name in ("times", "portfolio_pnl", "equity", "drawdown", "rolling_sharpe"):
        assert np.asarray(getattr(result, name)).reshape(-1).size == 400, name
    assert result.symbol == "XAUUSD"
    assert result.timeframe == "H1"
    assert result.periods_per_year == FOREX_XAUUSD.bars_per_year


def test_payload_is_valid_json_without_nan(tmp_path: Path) -> None:
    """落盘必须是**合法 JSON**：NaN / Infinity 会让 ``JSON.parse`` 直接抛错。"""
    result = run_full_backtest(TOKENS, _panel(n_bars=300), FOREX_XAUUSD, rolling_window=40)
    payload = to_payload(result)
    text = json.dumps(payload, ensure_ascii=False)
    assert "NaN" not in text and "Infinity" not in text

    target = tmp_path / "bt.json"
    target.write_text(text, encoding="utf-8")
    reloaded = json.loads(target.read_text(encoding="utf-8"))
    assert reloaded["meta"]["symbol"] == "XAUUSD"
    assert len(reloaded["series"]["equity"]) == 300
    # 滚动窗口未满的 warmup 段必须是 null
    assert reloaded["series"]["rolling_sharpe"][0] is None
    assert "val_score" not in json.dumps(reloaded["summary"])  # 绝不把搜索适应度当绩效


def test_payload_declares_caliber_and_truncates_trades() -> None:
    """meta 里必须带口径声明；明细超上限要截断并标记。"""
    result = run_full_backtest(TOKENS, _panel(n_bars=300), FOREX_XAUUSD, rolling_window=40)
    payload = to_payload(result, max_trades=2)
    assert "逐 bar 净收益" in payload["meta"]["caliber"]
    assert len(payload["trades"]) <= 2
    assert payload["trades_truncated"] == (len(result.trade_detail) > 2)
    # 相对/绝对回撤是两个不同单位，必须都给出来
    assert "max_drawdown" in payload["summary"]
    assert "max_drawdown_abs" in payload["summary"]
