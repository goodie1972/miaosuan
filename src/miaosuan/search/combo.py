# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""多因子组合评估与门禁（combine，T3）—— ``docs/combo-impl-spec.md`` §1/§3。

职责边界：本模块只做「**给定已选定的成员因子** → 组合信号 → 同款门禁 → 带结论候选」，
切分 / 数据加载 / 成员校验 / spec 组装都在 :mod:`miaosuan.pipeline` 里做（与 ``mine`` 对称）。

三条来自规格的硬约束：

* **§1.4 管线顺序**：成员因子（已 z-score）→ 符号对齐 → 加权和 → 因果重标定
  → **单次** tanh → **单次** 中性带（底层复用 :mod:`miaosuan.core.combo`，不重写）；
* **§1.7 门禁完全同款**：``evaluate_cost_curve → deflated_sharpe_ratio →
  evaluate_verdict``，与 ``search/mine._gate_candidate`` **同序同参**；折分打分走
  :meth:`AMFitnessEvaluator.fold_scores_of`（单因子 ``fold_scores`` 也转调它，
  单一实现，杜绝口径漂移）；
* **§1.3 n_trials 累加**：``n_trials_combo = Σ(各成员来源运行的 n_trials，按运行去重)
  + 权重搜索次数``。**等权与 IC 加权都计 0 次权重搜索** —— 二者都是确定性映射
  （无搜索自由度），符合 §1.2「禁止 GA 搜权重」。调用方传入已去重求和的 ``n_trials``。

与单因子的唯一**记录在案**的口径差异：组合体不扣「重复惩罚」。该惩罚是搜索期针对
**单条公式内部相邻重复 token** 的正则（AM ``_repetition_penalty``），组合体是若干条
已各自受过惩罚、且已跨文件去重的公式的线性混合，不存在该病理；强行在拼接序列上扣
罚只会让惩罚量随成员数任意增长。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import kurtosis as _kurtosis
from scipy.stats import skew as _skew

from ..core.combo import combine_factors, combo_to_position, sign_align
from ..core.signal import MIN_TRADE_EXPOSURE
from ..core.vm import DEFAULT_ROLL_WINDOW
from ..errors import ConfigError
from ..report.metrics import information_coefficient
from .ga import AMFitnessEvaluator

__all__ = [
    "COMBO_METHODS",
    "ComboCandidate",
    "combine",
    "weights_for",
]

#: 权重方法白名单：等权（零自由参数）/ IC 确定性加权。**禁止 GA 搜权重**（§1.2）。
COMBO_METHODS: tuple[str, ...] = ("equal", "ic")


@dataclass
class ComboCandidate:
    """组合候选：成员 + 权重 + 同款门禁结论（规格 §3 字段）。

    :param members: 成员因子的 RPN token 序列（已跨文件去重，顺序即加权顺序）。
    :param weights: 与 ``members`` 等长的确定性权重。
    :param val_score: 开发集验证分 = 组合信号逐折 IC 门控后的验证分均值。
    :param dsr: 去膨胀夏普（``n_trials`` 用 **Σ 累加值**，§1.3）。
    :param sharpe_2x: 2x 成本档 Sharpe（门禁硬下界输入）。
    :param wf_win_rate: Walk-Forward 折验证分为正的占比。
    :param verdict: 门禁结论（``DEPLOYABLE`` / ``RESEARCH_ONLY`` / ``BLOCKED``）。
    :param verdict_snapshot: 确定性 verdict 快照（JSON 字符串）。
    :param n_trials: 参与多重检验的试验数（Σ，见模块 docstring）。
    :param sharpes: 各成本档 Sharpe（喂给 ``Evidence.cost_sensitivity``）。
    :param train_score: 开发集训练分（同上折分口径，供审计）。
    """

    members: tuple[tuple[int, ...], ...]
    weights: tuple[float, ...]
    val_score: float
    dsr: float
    sharpe_2x: float
    wf_win_rate: float
    verdict: str
    verdict_snapshot: str
    n_trials: int
    sharpes: dict[float, float] = field(default_factory=dict)
    train_score: float = 0.0

    @property
    def n_members(self) -> int:
        """成员个数。"""
        return len(self.members)


def weights_for(
    ics: Sequence[float],
    method: str = "equal",
) -> np.ndarray:
    """按确定性方法算权重（§1.2：**无搜索自由度**）。

    Args:
        ics: 各成员的 IC（仅 ``method='ic'`` 用其绝对值）。
        method: ``"equal"`` → 等权 ``1/M``；``"ic"`` → ``|IC|`` 归一。

    Returns:
        和为 1 的 float64 权重向量。

    Raises:
        ConfigError: 方法未知；或 ``ic`` 方法下全部 ``|IC|`` 为 0（无信号可分权重）。
    """
    if method not in COMBO_METHODS:
        raise ConfigError(
            f"未知权重方法 {method!r}（可选 {list(COMBO_METHODS)}；禁止 GA 搜权重）",
            context={"method": method},
        )
    m = len(ics)
    if m == 0:
        raise ConfigError("成员为空，无法计算权重")
    if method == "equal":
        return np.full(m, 1.0 / m, dtype=np.float64)

    magnitudes = np.abs(np.asarray(ics, dtype=np.float64))
    total = float(magnitudes.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ConfigError(
            "method='ic' 需要至少一个成员 |IC| > 0（当前全为 0，无法按 IC 分权重）"
            "——请改用 --method equal",
            context={"n_members": m},
        )
    return magnitudes / total


def combine(
    members: Sequence[Sequence[int]],
    *,
    evaluator: AMFitnessEvaluator,
    n_trials: int,
    method: str = "equal",
    weights: np.ndarray | Sequence[float] | None = None,
    roll_window: int = DEFAULT_ROLL_WINDOW,
    neutral_band: float = MIN_TRADE_EXPOSURE,
    long_short: bool = True,
) -> ComboCandidate:
    """组合评估 + 门禁（规格 §1.4 管线 / §1.7 同款门禁）。

    Args:
        members: 成员因子的 RPN token 序列（调用方已跨文件去重；至少 1 个）。
        evaluator: 与单因子同款的适应度评估器（提供 dev 区特征、折分、成本、年化）。
        n_trials: **已按运行去重求和**的试验数（§1.3），本函数只用不改。
        method: 权重方法（``equal`` / ``ic``），仅在 ``weights`` 为 ``None`` 时生效。
        weights: 显式权重（给了就用，压过 ``method``）。
        roll_window: 因果重标定窗口（默认 500，与单因子栈同值）。
        neutral_band: 中性带（默认 0.05，与 ``Semantics.neutral_band`` 同值）。
        long_short: 是否多空双向（``False`` 即 ``long_only``）。

    Returns:
        带门禁结论的 :class:`ComboCandidate`。

    Raises:
        ConfigError: 成员为空、某成员在 dev 区不可求值、折分打分为空、权重非法。
    """
    if not members:
        raise ConfigError("成员为空：至少需要 1 个可组合的因子")

    factors: list[np.ndarray] = []
    for i, tokens in enumerate(members):
        got = evaluator.vm.execute([int(t) for t in tokens], evaluator.features)
        if got is None:
            raise ConfigError(
                f"成员 #{i} 在当前数据/词表下不可求值——请确认候选来自同一词表版本",
                context={"member": [int(t) for t in tokens]},
            )
        factors.append(got)

    # IC：符号对齐与 ic 加权共用同一份（同一 dev 区、同一实现）。
    ics = [
        float(information_coefficient(f, evaluator.target_ret)) for f in factors
    ]
    w = (
        np.asarray(weights, dtype=np.float64).reshape(-1)
        if weights is not None
        else weights_for(ics, method)
    )
    if w.size != len(members):
        raise ConfigError(
            f"weights 数量与成员数不一致：{w.size} vs {len(members)}",
            context={"weights": [float(x) for x in w]},
        )

    # ── §1.4 管线：对齐 → 加权和 → 因果重标定 → 单次 tanh → 单次中性带 ──
    aligned = sign_align(factors, ics)
    agg = combine_factors(list(aligned), w, roll_window=roll_window)
    position = combo_to_position(
        agg, neutral_band=neutral_band, long_only=not long_short
    )

    # ── 折分打分（与单因子同一个方法，IC 门控同款）────────────────────────
    train_scores, val_scores = evaluator.fold_scores_of(agg)
    if not val_scores:
        raise ConfigError(
            "组合信号在任何折上都拿不到验证分——请检查数据长度是否够 5 折切分",
            context={"n_folds": len(evaluator.folds)},
        )
    val_score = float(np.mean(val_scores))
    train_score = float(np.mean(train_scores))
    wf_win_rate = float(np.mean([1.0 if v > 0 else 0.0 for v in val_scores]))

    # ── §1.7 门禁：与 mine._gate_candidate 同序同参 ──────────────────────
    from ..gate.cost_curve import evaluate_cost_curve
    from ..gate.multiple_testing import deflated_sharpe_ratio
    from ..gate.verdict import evaluate_verdict

    oos = [slice(f["val_start"], f["val_end"]) for f in evaluator.folds]
    position_oos = np.concatenate([position[:, sl] for sl in oos], axis=1)
    ret_oos = np.concatenate(
        [np.asarray(evaluator.target_ret[:, sl], dtype=np.float64) for sl in oos],
        axis=1,
    )
    curve = evaluate_cost_curve(
        position_oos,
        ret_oos,
        base_cost_rate=evaluator.cost_rate,
        periods_per_year=evaluator.periods_per_year,
    )
    sharpes = curve.sharpes()
    sharpe_1x = curve.sharpe_at(1.0)
    sharpe_2x = curve.sharpe_at(2.0)

    prev = np.roll(position_oos, 1, axis=1)
    prev[:, 0] = 0.0
    turnover = np.abs(position_oos - prev)
    pnl = position_oos * ret_oos - turnover * float(evaluator.cost_rate)
    flat = pnl.reshape(-1)
    std = float(flat.std())
    sharpe_bar = float(flat.mean() / (std + 1e-12)) if std > 0 else 0.0
    dsr = float(
        deflated_sharpe_ratio(
            sharpe_bar,
            n_trials=max(1, int(n_trials)),
            n_obs=int(flat.size),
            skew=float(_skew(flat)) if flat.size > 2 else 0.0,
            kurtosis=float(_kurtosis(flat)) if flat.size > 3 else 3.0,
        ).dsr
    )

    verdict = evaluate_verdict(
        val_score=val_score,
        train_score=train_score,
        dsr=dsr,
        sharpe_1x=sharpe_1x,
        sharpe_2x=sharpe_2x,
        wf_win_rate=wf_win_rate,
        n_trials=max(1, int(n_trials)),
        n_obs=int(np.asarray(evaluator.target_ret).shape[1]),
    )
    return ComboCandidate(
        members=tuple(tuple(int(t) for t in m) for m in members),
        weights=tuple(float(x) for x in w),
        val_score=val_score,
        dsr=dsr,
        sharpe_2x=sharpe_2x,
        wf_win_rate=wf_win_rate,
        verdict=str(verdict.status),
        verdict_snapshot=verdict.snapshot(),
        n_trials=int(n_trials),
        sharpes=dict(sharpes),
        train_score=train_score,
    )
