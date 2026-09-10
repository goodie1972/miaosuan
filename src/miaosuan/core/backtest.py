"""向量化组合回测评估器（numpy 化移植，M5）。

移植自冻结 AlphaMaster 的 ``model_core/backtest.py``（原 torch 实现），逐行对齐：

* :func:`estimate_periods_per_year` —— 由时间戳估计「每年 bar 数」（自动年化）；
* :class:`MT5Backtest` —— 组合级多目标 Reward（Sortino / Calmar / 时序 IC / 品种一致性 /
  成本压力 / 换手质量 / Beta 中性 / 前后一致性）。

关键等价与参数化：

* ``torch.roll(x, 1, dims=1)`` → ``np.roll(x, 1, axis=1)``；``torch.cummax`` →
  ``np.maximum.accumulate``；``.std(unbiased=False)`` → ``np.std(..., ddof=0)``。
* AM 由 ``strategy_manager.signal`` 取仓位 → 妙算改用同包 :mod:`.signal`。
* AM 由根目录 ``config.Config.COST_RATE``（=0.0003）/ ``ModelConfig.REWARD_MODE``
  （当前 ``"ftmo"``）隐式决定行为；妙算改为**参数注入** ``cost_rate``（默认 0.0003）与
  ``reward_mode``（默认 ``"ftmo"``），默认值与 AM 一致以保证对拍，``core/`` 不读环境/不做 IO。
* 数据驱动年化因子 ``estimate_periods_per_year`` 保留；AM 的 tqdm/print 兜底告警改为
  ``warnings.warn``（无 IO 副作用）。

依赖方向：``core/backtest.py`` 仅依赖 :mod:`.signal`，不 import 任何上层包。
"""

from __future__ import annotations

import math
import warnings

import numpy as np

from .signal import compute_target_positions_stateless

__all__ = ["MT5Backtest", "estimate_periods_per_year"]

_H1_PERIODS_PER_YEAR = 6240
_SORTINO_CLIP = 20.0

_SECONDS_PER_YEAR = 365.25 * 86400.0

# AM 兜底成本率（与生产 run_backtest.py 一致：手续费 0.02% + 滑点 0.01%）。
_DEFAULT_COST_RATE = 0.0003
# AM ``ModelConfig.REWARD_MODE`` 当前取值（对拍基准）。
_DEFAULT_REWARD_MODE = "ftmo"


def estimate_periods_per_year(times: np.ndarray) -> int:
    """从时间戳序列估计「每年 bar 数」（年化因子）。

    用 ``T / years_span``，``years_span = (t_last - t_first) / SECONDS_PER_YEAR``。该方法
    自动适应不同市场（A 股 / 外汇 / 加密）与周期——数据里只含交易时段的 bar，跨日历年的
    bar 密度天然反映该市场交易频率，无需按周期/市场写死。

    Args:
        times: ``[N, T]`` 或 ``[T]`` 的 Unix 秒时间戳（numpy 数组）。

    Returns:
        int，每年 bar 数；数据不足时回退到 ``_H1_PERIODS_PER_YEAR``。
    """
    arr = np.asarray(times).astype(np.float64)

    def _estimate_one(row: np.ndarray, t_count: int) -> int | None:
        if t_count < 2:
            return None
        span = float(row[-1] - row[0])
        if span <= 0:
            return None
        years = span / _SECONDS_PER_YEAR
        if years <= 0:
            return None
        ppy = t_count / years
        return int(max(10, min(600000, round(ppy))))

    if arr.ndim == 2:
        # 多品种时按各品种独立估计后取中位数，避免单一品种异常值影响。
        estimates: list[int] = []
        for r in range(arr.shape[0]):
            est = _estimate_one(arr[r], arr.shape[1])
            if est is not None:
                estimates.append(est)
        if not estimates:
            warnings.warn(
                "estimate_periods_per_year: 所有品种时间戳均无效，回退到默认 H1=6240。"
                "请检查 raw_dict['time'] 是否正确。",
                RuntimeWarning,
                stacklevel=2,
            )
            return _H1_PERIODS_PER_YEAR
        estimates.sort()
        return estimates[len(estimates) // 2]
    if arr.ndim == 1:
        est = _estimate_one(arr, arr.shape[0])
        if est is None:
            warnings.warn(
                "estimate_periods_per_year: 时间戳无效或不足，回退到默认 H1=6240。"
                "请检查 raw_dict['time'] 是否正确。",
                RuntimeWarning,
                stacklevel=2,
            )
            return _H1_PERIODS_PER_YEAR
        return est
    warnings.warn(
        f"estimate_periods_per_year: 时间戳维度异常 ndim={arr.ndim}，回退到默认 H1=6240。",
        RuntimeWarning,
        stacklevel=2,
    )
    return _H1_PERIODS_PER_YEAR


class MT5Backtest:
    """MT5 组合级回测评估器（AM ``MT5Backtest`` 的 numpy 移植，行为等价）。"""

    def __init__(
        self,
        cost_rate: float | None = None,
        periods_per_year: int = _H1_PERIODS_PER_YEAR,
        reward_mode: str = _DEFAULT_REWARD_MODE,
    ) -> None:
        # 成本率与奖励模式改为参数注入（默认值与 AM 一致：0.0003 / "ftmo"）。
        self.cost_rate = _DEFAULT_COST_RATE if cost_rate is None else float(cost_rate)
        self.periods_per_year = int(periods_per_year)
        self.reward_mode = str(reward_mode)

    # ── 基础统计 ──────────────────────────────────────────────────────────

    def _sortino(self, pnl: np.ndarray, eps: float = 1e-8) -> float:
        flat = pnl.reshape(-1)
        mean_pnl = float(flat.mean())
        downside = flat[flat < 0]
        raw_std = float(downside.std(ddof=0)) if downside.size > 0 else 0.0
        # 下行标准差地板改为全序列 std 的 20%，防稀疏 PnL 靠极小分母刷高分。
        full_std = max(float(flat.std(ddof=0)), eps)
        floor = max(full_std * 0.2, eps)
        downside_std = max(raw_std, floor)
        sortino = mean_pnl / downside_std * math.sqrt(self.periods_per_year)
        return float(min(max(sortino, -_SORTINO_CLIP), _SORTINO_CLIP))

    def _calmar(self, pnl: np.ndarray, eps: float = 1e-8) -> float:
        """Calmar = annualized_return / max_drawdown（截断到 [-10, 10]）。"""
        flat = pnl.reshape(-1)
        ann_ret = float(flat.mean()) * self.periods_per_year
        cum = np.cumsum(flat)
        peak = np.maximum.accumulate(cum)
        drawdown = float((peak - cum).max())
        drawdown = max(drawdown, eps)
        calmar = ann_ret / drawdown
        return float(min(max(calmar, -10.0), 10.0))

    # ── 组合级评分组件 ────────────────────────────────────────────────────

    def _ts_ic_stability(self, factors: np.ndarray, target_ret: np.ndarray) -> float:
        """时序 IC 稳定性：每品种内部 ``factor[t]`` 与 ``ret[t+1]`` 相关性的均值。"""
        n, t = factors.shape
        if t < 10:
            return 0.0

        ic_list: list[float] = []
        for i in range(n):
            x = factors[i, :-1]
            y = target_ret[i, 1:]
            xm = x - x.mean()
            ym = y - y.mean()
            sx = float(np.sqrt((xm**2).mean()))
            sy = float(np.sqrt((ym**2).mean()))
            if sx < 1e-6 or sy < 1e-6:
                continue
            ic = float((xm * ym).mean()) / (sx * sy + 1e-8)
            ic_list.append(ic)

        if not ic_list:
            return 0.0

        ic_mean = sum(ic_list) / len(ic_list)
        ic_std = (sum((v - ic_mean) ** 2 for v in ic_list) / len(ic_list)) ** 0.5
        stability = ic_mean / (ic_std + 1e-6)
        return float(max(-3.0, min(3.0, stability)))

    def _symbol_consistency(
        self,
        per_symbol_sortino: list[float],
        per_symbol_trade_count: list[int] | None = None,
        eval_bars: int = 0,
    ) -> float:
        """品种一致性惩罚/奖励（规则见 AM 文档）。"""
        n = len(per_symbol_sortino)
        if n == 0:
            return 0.0

        min_trades = max(5, eval_bars // 100) if eval_bars > 0 else 5

        if per_symbol_trade_count is not None:
            n_inactive = sum(1 for c in per_symbol_trade_count if c < min_trades)
            if n_inactive / n > 0.4:
                return -3.0

        if any(s < -2.0 for s in per_symbol_sortino):
            return -2.0

        if per_symbol_trade_count is not None:
            active_sortinos = [
                s
                for s, c in zip(per_symbol_sortino, per_symbol_trade_count, strict=False)
                if c >= min_trades
            ]
        else:
            active_sortinos = per_symbol_sortino

        if not active_sortinos:
            return -3.0

        n_positive = sum(1 for s in active_sortinos if s > 0)
        ratio = n_positive / len(active_sortinos)

        score = (ratio - 0.6) / 0.6 * 1.0 if ratio < 0.6 else (ratio - 0.6) / 0.4 * 1.0

        if ratio == 1.0:
            score += 0.5

        return float(score)

    def _cost_stress(
        self,
        position: np.ndarray,
        target_ret: np.ndarray,
        stress_mult: float = 2.0,
    ) -> float:
        """成本压力测试：2 倍成本下的 Sortino（截断到 [-5, 5]）。"""
        prev_pos = np.roll(position, 1, axis=1)
        prev_pos[:, 0] = 0.0
        turnover = np.abs(position - prev_pos)
        stressed_pnl = position * target_ret - turnover * self.cost_rate * stress_mult
        sortino = self._sortino(stressed_pnl)
        return float(min(max(sortino, -5.0), 5.0))

    def _turnover_quality(self, position: np.ndarray) -> float:
        """交易频率质量奖励（每天约 1 笔为最优，目标每 12 bar 一笔）。"""
        n, t = position.shape
        pos_2d = np.asarray(position).tolist()
        all_runs: list[int] = []
        total_trades = 0

        for i in range(n):
            runs: list[int] = []
            cur_len = 0
            cur_dir = 0
            for p in pos_2d[i]:
                pi = int(p)
                if pi != 0:
                    if pi == cur_dir:
                        cur_len += 1
                    else:
                        if cur_len > 0:
                            runs.append(cur_len)
                        cur_dir, cur_len = pi, 1
                else:
                    if cur_len > 0:
                        runs.append(cur_len)
                    cur_dir, cur_len = 0, 0
            if cur_len > 0:
                runs.append(cur_len)
            all_runs.extend(runs)
            total_trades += len(runs)

        total_bars = n * t
        target_trades = total_bars / 12.0
        actual_ratio = total_trades / max(target_trades, 1.0)

        if actual_ratio <= 0:
            freq_score = -2.0
        elif actual_ratio < 0.05:
            freq_score = -2.0 + actual_ratio / 0.05
        elif actual_ratio < 0.5:
            freq_score = -1.0 + (actual_ratio - 0.05) / 0.45
        elif actual_ratio <= 2.0:
            log_r = math.log(actual_ratio) / math.log(2.0)
            freq_score = 1.0 * math.exp(-0.5 * log_r**2)
        elif actual_ratio <= 8.0:
            freq_score = 0.5 - (actual_ratio - 2.0) / 6.0 * 1.5
        else:
            freq_score = -2.0

        hold_bonus = 0.0
        if all_runs:
            avg_hold = sum(all_runs) / len(all_runs)
            hold_bonus = min(0.3, math.log(max(avg_hold, 1.0)) / math.log(30.0) * 0.3)

        return float(freq_score + hold_bonus)

    def _beta_neutral_penalty(self, position: np.ndarray) -> float:
        """Beta 中性惩罚：多空比例严重失衡时扣分。"""
        flat = position.reshape(-1)
        long_ratio = float((flat > 0.05).mean())
        short_ratio = float((flat < -0.05).mean())
        max_ratio = max(long_ratio, short_ratio)
        if max_ratio > 0.85:
            excess = (max_ratio - 0.85) / 0.15
            return -2.0 * excess
        if max_ratio > 0.70:
            excess = (max_ratio - 0.70) / 0.15
            return -0.5 * excess
        return 0.0

    def _half_consistency_bonus(self, pnl: np.ndarray) -> float:
        """前后一致性奖励：前半段和后半段 Sortino 同号时加分。"""
        t = pnl.shape[1]
        if t < 20:
            return 0.0
        half = t // 2
        s1 = self._sortino(pnl[:, :half])
        s2 = self._sortino(pnl[:, half:])
        if s1 > 0 and s2 > 0:
            return 0.5
        if s1 * s2 < 0:
            return -1.0
        return 0.0

    def _exposure_penalty(self, position: np.ndarray) -> float:
        """在场时间惩罚（仅下限，无上限）：只惩罚极稀疏交易（<10% 在场）。"""
        flat = np.abs(position.reshape(-1))
        exposure = float(flat.mean())
        if exposure < 0.10:
            return float((exposure / 0.10 - 1.0) * 2.0)
        return 0.0

    def _turnover_penalty(self, turnover: np.ndarray) -> float:
        """梯度式换手率惩罚。"""
        mean_to = float(turnover.mean())
        penalty = float(min(max((mean_to - 0.2) * 3.0, 0.0), 3.0))
        return -penalty

    # ── Walk-Forward 辅助接口 ─────────────────────────────────────────────

    def evaluate_fold(
        self,
        factors: np.ndarray,
        target_ret: np.ndarray,
        train_start: int,
        train_end: int,
        val_start: int,
        val_end: int,
    ) -> tuple[float, float]:
        """在指定训练/验证切片上计算组合多目标得分（train_score, val_score）。"""
        position = compute_target_positions_stateless(factors)

        prev_pos = np.roll(position, 1, axis=1)
        prev_pos[:, 0] = 0.0
        turnover = np.abs(position - prev_pos)
        pnl = position * target_ret - turnover * self.cost_rate

        pnl_train = pnl[:, train_start:train_end]
        pnl_val = pnl[:, val_start:val_end]

        train_bars = train_end - train_start
        train_score = self._multi_objective(
            factors[:, train_start:train_end],
            target_ret[:, train_start:train_end],
            pnl_train,
            position[:, train_start:train_end],
            eval_bars=train_bars,
        ) + self._turnover_penalty(turnover[:, train_start:train_end])

        val_bars = val_end - val_start
        base_val = self._multi_objective(
            factors[:, val_start:val_end],
            target_ret[:, val_start:val_end],
            pnl_val,
            position[:, val_start:val_end],
            eval_bars=val_bars,
        )
        oos_sor = self._sortino(pnl_val)
        mult = max(0.1, 0.5 + oos_sor * 0.4) if oos_sor <= 0 else min(1.2, 1.0 + oos_sor * 0.1)
        val_score = base_val * mult

        return float(train_score), float(val_score)

    def _reversal_bonus(self, factors: np.ndarray) -> float:
        """反转奖励：鼓励因子有低/负自相关（均值回归特征）。"""
        n = factors.shape[0]
        scores: list[float] = []
        for i in range(n):
            x = factors[i, :-1]
            y = factors[i, 1:]
            xm = x - x.mean()
            ym = y - y.mean()
            sx = float(np.sqrt((xm**2).mean()))
            sy = float(np.sqrt((ym**2).mean()))
            ac1 = (
                float((xm * ym).mean()) / (sx * sy + 1e-8)
                if sx > 1e-6 and sy > 1e-6
                else 0.0
            )
            bonus = 1.0 - abs(ac1)
            if ac1 < 0:
                bonus = bonus + 0.5
            bonus = float(min(max(bonus, -1.0), 2.0))
            scores.append(bonus)
        return float(sum(scores) / len(scores))

    def _symmetry_check(self, position: np.ndarray) -> float:
        """多空对称性检查：奖励 50/50 多空分布。"""
        long_ratio = float((position > 0).mean())
        short_ratio = float((position < 0).mean())
        deviation = abs(long_ratio - 0.5) + abs(short_ratio - 0.5)
        bonus = 1.0 - 2.0 * deviation
        return float(min(max(bonus, -1.0), 1.0))

    def _multi_objective(
        self,
        factors: np.ndarray,
        target_ret: np.ndarray,
        pnl: np.ndarray,
        position: np.ndarray,
        eval_bars: int = 0,
    ) -> float:
        """收益优先的多目标评分（AM ``_multi_objective`` 逐分支等价移植）。"""
        n = pnl.shape[0]

        # ── 绝对收益（年化 log return）──────────────────────────────────
        ann_ret = float(pnl.mean()) * self.periods_per_year

        port_sortino = self._sortino(pnl)
        port_calmar = self._calmar(pnl)
        ts_ic = self._ts_ic_stability(factors, target_ret)
        tq = self._turnover_quality(position)
        exp_pen = self._exposure_penalty(position)

        if n == 1:
            beta_pen = self._beta_neutral_penalty(position)
            consist = self._half_consistency_bonus(pnl)

            if self.reward_mode == "forex":
                rev_bonus = self._reversal_bonus(factors)
                sym_bonus = self._symmetry_check(position)
                return (
                    0.25 * ann_ret
                    + 0.05 * port_sortino
                    + 0.05 * port_calmar
                    + 0.25 * ts_ic
                    + 0.20 * rev_bonus
                    + 0.15 * sym_bonus
                    + 0.05 * tq
                    + exp_pen
                    + beta_pen
                    + consist
                )

            if self.reward_mode == "ftmo":
                return (
                    0.80 * ann_ret
                    + 0.05 * port_sortino
                    + 0.10 * port_calmar
                    + 0.03 * ts_ic
                    + 0.02 * tq
                    + exp_pen
                    + beta_pen
                    + consist
                )
            return (
                0.60 * ann_ret
                + 0.15 * port_sortino
                + 0.10 * port_calmar
                + 0.10 * ts_ic
                + 0.05 * tq
                + exp_pen
                + beta_pen
                + consist
            )

        per_sym_sortino: list[float] = []
        per_sym_trade_count: list[int] = []
        for i in range(n):
            per_sym_sortino.append(self._sortino(pnl[i]))
            pos_n = np.abs(position[i])
            diff = np.abs(pos_n[1:] - pos_n[:-1])
            trades = int((diff > 0.1).sum())
            per_sym_trade_count.append(trades)

        sym_cons = self._symbol_consistency(
            per_sym_sortino, per_sym_trade_count, eval_bars=eval_bars
        )
        cost_s = self._cost_stress(position, target_ret)
        beta_pen = self._beta_neutral_penalty(position)
        consist = self._half_consistency_bonus(pnl)

        if self.reward_mode == "ftmo":
            return (
                0.75 * ann_ret
                + 0.05 * port_sortino
                + 0.10 * port_calmar
                + 0.02 * ts_ic
                + 0.03 * sym_cons
                + 0.02 * cost_s
                + 0.03 * tq
                + exp_pen
                + beta_pen
                + consist
            )

        return (
            0.60 * ann_ret
            + 0.10 * port_sortino
            + 0.05 * port_calmar
            + 0.10 * ts_ic
            + 0.05 * sym_cons
            + 0.05 * cost_s
            + 0.05 * tq
            + exp_pen
            + beta_pen
            + consist
        )

    # ── 公开接口（非 Walk-Forward 模式）───────────────────────────────────

    def evaluate(
        self,
        factors: np.ndarray,
        raw_dict: dict[str, object],
        target_ret: np.ndarray,
    ) -> tuple[float, float]:
        """评估一组 Alpha 因子（含 OOS 80/20 门控），返回 ``(score, mean_oos)``。"""
        position = compute_target_positions_stateless(factors)

        prev_pos = np.roll(position, 1, axis=1)
        prev_pos[:, 0] = 0.0
        turnover = np.abs(position - prev_pos)
        pnl = position * target_ret - turnover * self.cost_rate

        t = factors.shape[1]
        split = int(math.floor(t * 0.8))

        score = self._multi_objective(
            factors[:, :split],
            target_ret[:, :split],
            pnl[:, :split],
            position[:, :split],
            eval_bars=split,
        ) + self._turnover_penalty(turnover[:, :split])

        # OOS 门控（最后 20%）
        pnl_oos = pnl[:, split:]
        oos_sor = self._sortino(pnl_oos)
        if oos_sor <= 0:
            mult = max(0.1, 0.5 + oos_sor * 0.4)
            score = score * mult
        else:
            score = score * min(1.2, 1.0 + oos_sor * 0.1)

        mean_oos = float(pnl_oos.mean())
        return float(score), mean_oos
