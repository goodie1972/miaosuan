# -*- coding: utf-8 -*-
# Copyright (c) 2026 Original Author
# Copyright (c) 2026 Your Company
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""``mine`` 端到端编排：数据 → 切分 → 搜索 → 门禁 → 带评分候选（架构 §6，验收 #1）。

这是把 M6–M12 串成一条完整流水线的**唯一编排点**：

1. **切分**：:func:`~miaosuan.data.split.make_split` 三段切分并**一次性封印 hold-out**；
2. **开发区**：搜索只在 ``[0, val_end)``（train ∪ purge ∪ val）上进行 —— **hold-out 结构性地
   被排除在一切搜索/评估之外**（年化因子取自 profile ``bars_per_year``，不硬编码）；
3. **搜索**：:class:`~miaosuan.search.ga.RpnGA` 在三档预算内进化；
4. **门禁**：对最终种群 top-k 逐一执行 M12 综合判定（DSR + 衰减比 + 2x 成本 Sharpe + WF 胜率）；
5. **产出**：带评分与 verdict 的候选公式列表 + 可复现元信息。

**搜索路径不触碰 hold-out** 由结构保证（开发区切片止于 ``val_end``）+ 测试断言（见
``tests/test_search_mine.py``：hold-out 指纹封印**恰一次**、评估器所见 bar 数 < 全量 bar 数）。

依赖方向：``search/mine.py`` 依赖 ``core`` / ``data`` / ``market`` / ``report`` / ``gate`` 与同包
``search``。**唯一** ``search → gate`` 边即在此；``gate`` 不反向依赖 ``search``（无环）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import kurtosis as _kurtosis
from scipy.stats import skew as _skew

from ..config import AppConfig
from ..core.features import compute_features
from ..core.signal import compute_target_positions_stateless
from ..data.panel import Panel
from ..data.split import DEFAULT_SEAL_PATH, DataSplit, HoldoutSealRegistry, make_split
from ..market.profiles import FrozenMarketProfile, get_profile
from .budget import DEFAULT_PROFILE, Budget
from .ga import AMFitnessEvaluator, FormulaEvaluator, RpnGA
from .rpn import Individual, decode

__all__ = [
    "Candidate",
    "MineResult",
    "compute_target_ret",
    "mine",
]

_logger = logging.getLogger("miaosuan.search.mine")

#: 默认取用的候选数。
DEFAULT_TOP_K: int = 5


def compute_target_ret(open_arr: np.ndarray) -> np.ndarray:
    """计算前瞻收益 ``target_ret[t] = log(open[t+2] / open[t+1])``（AM ``data_manager`` 等价）。

    末 2 列无未来收益 → 置 0；``open`` 分母为 0 时以 1 兜底（防止 ``log(inf)``）。
    """
    open_arr = np.asarray(open_arr, dtype=np.float64)
    n, t = open_arr.shape
    out = np.zeros((n, t), dtype=np.float32)
    if t >= 3:
        num = open_arr[:, 2:]
        den = open_arr[:, 1:-1].copy()
        den[den == 0] = 1.0
        out[:, : t - 2] = np.log(num / den)
    return out


@dataclass
class Candidate:
    """带评分与门禁结论的候选公式。

    :param tokens: 定长 token 元组。
    :param decoded: 人类可读公式。
    :param train_score: 开发集训练分。
    :param val_score: 开发集验证分（GA 目标）。
    :param status: 评估状态（``ok`` / ``none`` / ``const``）。
    :param sharpes: 成本敏感性各档 Sharpe ``{倍率: 值}``。
    :param sharpe_2x: 2x 成本 Sharpe。
    :param wf_win_rate: Walk-Forward 折验证分为正的占比。
    :param dsr: Deflated Sharpe 显著性概率。
    :param verdict: 门禁综合结论（``DEPLOYABLE`` / ``RESEARCH_ONLY`` / ``BLOCKED``）。
    :param verdict_snapshot: 确定性 verdict 快照（JSON 字符串）。
    """

    tokens: tuple[int, ...]
    decoded: str
    train_score: float
    val_score: float
    status: str
    sharpes: dict[float, float]
    sharpe_2x: float
    wf_win_rate: float
    dsr: float
    verdict: str
    verdict_snapshot: str

    @property
    def is_deployable(self) -> bool:
        """门禁是否判为可上线。"""
        return self.verdict == "DEPLOYABLE"


@dataclass
class MineResult:
    """``mine`` 完整结果（含可审计的搜索/门禁证据）。

    :param candidates: 带评分与 verdict 的候选（按适应度降序）。
    :param best: 最优候选。
    :param split: 三段切分结果（hold-out 已封印）。
    :param stop_reason: 搜索停止原因（``WALL_CLOCK`` / ``EARLY_STOP`` / ``MAX_GENERATIONS`` / ``CONVERGED``）。
    :param generations: 实际完成代数。
    :param n_evaluations: 真实评估次数（缓存未命中）。
    :param initial_diversity: 初始平均两两汉明距离。
    :param min_diversity: 全程最低平均两两汉明距离。
    :param dev_bars: 搜索使用的开发区 bar 数（``< n_bars`` 证明未触碰 hold-out）。
    :param n_bars: 全量 bar 数。
    :param wall_clock_seconds: 搜索墙钟耗时（秒）。
    :param history: 逐代统计快照。
    :param config_snapshot: 配置快照（可复现）。
    """

    candidates: list[Candidate]
    best: Candidate
    split: DataSplit
    stop_reason: str
    generations: int
    n_evaluations: int
    initial_diversity: float
    min_diversity: float
    dev_bars: int
    n_bars: int
    wall_clock_seconds: float
    history: list[object] = field(default_factory=list)
    config_snapshot: dict[str, object] = field(default_factory=dict)

    @property
    def diversity_ratio(self) -> float:
        """全程最低多样性 / 初始多样性（验收 #3：≥ 0.4）。"""
        if self.initial_diversity <= 0:
            return 1.0
        return self.min_diversity / self.initial_diversity

    @property
    def holdout_untouched(self) -> bool:
        """开发区严格小于全量（结构性证据：搜索未触碰 hold-out）。"""
        return self.dev_bars < self.n_bars


def _resolve_profile(panel: Panel, profile: FrozenMarketProfile | None) -> FrozenMarketProfile:
    """确定评估用市场画像（显式注入优先，否则按面板名查找）。"""
    if profile is not None:
        return profile
    name = panel.market_profile_name
    if not name:
        raise ValueError("panel.market_profile_name 为空且未显式提供 profile")
    return get_profile(name)


def _oos_slices(folds: list[dict[str, int]]) -> list[slice]:
    """把各折验证区间转成 ``slice`` 列表。"""
    return [slice(f["val_start"], f["val_end"]) for f in folds]


def _gate_candidate(
    ind: Individual,
    evaluator: AMFitnessEvaluator,
    *,
    base_cost_rate: float,
    periods_per_year: int,
    n_trials: int,
) -> Candidate:
    """对单个候选计算 M12 门禁证据并给出 verdict。"""
    factor = evaluator.vm.execute([int(t) for t in ind.tokens], evaluator.features)
    train_scores, val_scores = evaluator.fold_scores(ind.tokens)
    wf_win_rate = (
        float(np.mean([1.0 if v > 0 else 0.0 for v in val_scores])) if val_scores else 0.0
    )
    dsr = 0.0
    sharpe_1x = 0.0
    sharpe_2x = 0.0
    sharpes: dict[float, float] = {}
    if factor is not None:
        oos = _oos_slices(evaluator.folds)
        pos_parts = []
        ret_parts = []
        for sl in oos:
            pos_parts.append(compute_target_positions_stateless(factor[:, sl]))
            ret_parts.append(np.asarray(evaluator.target_ret[:, sl], dtype=np.float64))
        position = np.concatenate(pos_parts, axis=1)
        ret = np.concatenate(ret_parts, axis=1)

        from ..gate.cost_curve import evaluate_cost_curve
        from ..gate.multiple_testing import deflated_sharpe_ratio

        curve = evaluate_cost_curve(
            position,
            ret,
            base_cost_rate=base_cost_rate,
            periods_per_year=periods_per_year,
        )
        sharpes = curve.sharpes()
        sharpe_1x = curve.sharpe_at(1.0)
        sharpe_2x = curve.sharpe_at(2.0)

        prev = np.roll(position, 1, axis=1)
        prev[:, 0] = 0.0
        turnover = np.abs(position - prev)
        pnl = position * ret - turnover * float(base_cost_rate)
        flat = pnl.reshape(-1)
        std = float(flat.std())
        sharpe_bar = float(flat.mean() / (std + 1e-12)) if std > 0 else 0.0
        dsr_res = deflated_sharpe_ratio(
            sharpe_bar,
            n_trials=max(1, int(n_trials)),
            n_obs=int(flat.size),
            skew=float(_skew(flat)) if flat.size > 2 else 0.0,
            kurtosis=float(_kurtosis(flat)) if flat.size > 3 else 3.0,
        )
        dsr = float(dsr_res.dsr)

    from ..gate.verdict import evaluate_verdict

    verdict = evaluate_verdict(
        val_score=float(ind.val_score),
        train_score=float(ind.train_score),
        dsr=dsr,
        sharpe_1x=sharpe_1x,
        sharpe_2x=sharpe_2x,
        wf_win_rate=wf_win_rate,
        n_trials=max(1, int(n_trials)),
        n_obs=int(np.asarray(evaluator.target_ret).shape[1]),
    )
    return Candidate(
        tokens=ind.key(),
        decoded=decode(ind.tokens),
        train_score=float(ind.train_score),
        val_score=float(ind.val_score),
        status=ind.status,
        sharpes=sharpes,
        sharpe_2x=float(sharpe_2x),
        wf_win_rate=float(wf_win_rate),
        dsr=float(dsr),
        verdict=verdict.status,
        verdict_snapshot=verdict.snapshot(),
    )


def mine(
    panel: Panel,
    *,
    config: AppConfig,
    registry: HoldoutSealRegistry | None = None,
    profile: FrozenMarketProfile | None = None,
    budget: Budget | None = None,
    evaluator: FormulaEvaluator | None = None,
    top_k: int = DEFAULT_TOP_K,
    budget_profile: str = DEFAULT_PROFILE,
    n_folds: int = 5,
    gap: int = 20,
) -> MineResult:
    """端到端因子挖掘：数据 → 切分（封印 hold-out）→ 搜索 → 门禁 → 候选。

    :param panel: 全量行情面板（``market_profile_name`` 指向市场画像）。
    :param config: 顶层配置（切分 / 搜索 / 种子）。
    :param registry: hold-out 封印台账（缺省落盘 :data:`DEFAULT_SEAL_PATH`；测试用 ``IN_MEMORY``）。
    :param profile: 市场画像（缺省按 ``panel.market_profile_name`` 查找）。
    :param budget: 预算（缺省按 ``budget_profile`` 构造）；其 ``pop_size`` 等可按测试缩减。
    :param evaluator: 适应度评估器（缺省构造 :class:`AMFitnessEvaluator`；测试可注入合成评估器）。
    :param top_k: 汇总的候选数。
    :param budget_profile: 预算档位名（``quick`` / ``standard`` / ``deep``）。
    :param n_folds: Walk-Forward 折数（默认 5，与 AM 一致）。
    :param gap: 折间间隔（默认 20，与 AM ``WF_GAP`` 一致）。
    """
    prof = _resolve_profile(panel, profile)
    seal_registry = registry if registry is not None else HoldoutSealRegistry(DEFAULT_SEAL_PATH)

    # ① 三段切分 + 一次性封印 hold-out（搜索路径此后只看到 [0, val_end)）。
    split = make_split(panel, config.split, registry=seal_registry, seed=config.seed, seal=True)
    assert split.indices is not None
    val_end = split.indices.ranges()["val"][1]
    dev_panel = panel.slice_view(0, val_end)
    dev_bars = int(dev_panel.n_bars)

    # ② 开发区特征 + 前瞻收益（hold-out 结构性被排除）。
    features = compute_features(dev_panel.to_raw_dict())
    target_ret = compute_target_ret(dev_panel.open)

    # ③ 适应度评估器（年化因子取自 profile.bars_per_year，不硬编码）。
    eval_impl: FormulaEvaluator
    if evaluator is not None:
        eval_impl = evaluator
    else:
        eval_impl = AMFitnessEvaluator(
            features,
            target_ret,
            cost_rate=float(prof.cost_model.total("buy")),
            periods_per_year=int(prof.bars_per_year),
            n_folds=int(n_folds),
            gap=int(gap),
        )

    # ④ 搜索。
    active_budget = budget if budget is not None else Budget.from_profile(budget_profile)
    ga = RpnGA(config.search, seed=config.seed, length=config.search.formula_len)

    def _log_generation(stats: object) -> None:
        gen = getattr(stats, "generation", -1)
        best = getattr(stats, "best_fitness", float("nan"))
        div = getattr(stats, "diversity", float("nan"))
        _logger.info(
            "gen=%d best=%.4f diversity=%.3f evals=%d immigrants=%d",
            gen,
            best,
            div,
            getattr(stats, "n_evaluations", 0),
            getattr(stats, "n_immigrants", 0),
        )

    ga.run(active_budget, eval_impl, on_generation=_log_generation)
    result = ga.last_result
    assert result is not None  # run() 一定写入 last_result

    # ⑤ 门禁：对最终 top-k 逐一判定。
    n_trials = int(getattr(eval_impl, "n_evaluated", result.n_evaluations))
    base_cost_rate = float(prof.cost_model.total("buy"))
    candidates: list[Candidate] = []
    for ind in ga.top_individuals(top_k):
        if isinstance(eval_impl, AMFitnessEvaluator):
            candidates.append(
                _gate_candidate(
                    ind,
                    eval_impl,
                    base_cost_rate=base_cost_rate,
                    periods_per_year=int(prof.bars_per_year),
                    n_trials=n_trials,
                )
            )
        else:
            candidates.append(
                Candidate(
                    tokens=ind.key(),
                    decoded=decode(ind.tokens),
                    train_score=float(ind.train_score),
                    val_score=float(ind.val_score),
                    status=ind.status,
                    sharpes={},
                    sharpe_2x=0.0,
                    wf_win_rate=0.0,
                    dsr=0.0,
                    verdict="UNKNOWN",
                    verdict_snapshot="{}",
                )
            )
    if not candidates:  # pragma: no cover - 空种群不应发生（初始化保证非空）
        raise RuntimeError("mine: 未产出任何候选（种群为空？）")

    return MineResult(
        candidates=candidates,
        best=candidates[0],
        split=split,
        stop_reason=result.stop_reason,
        generations=result.generations,
        n_evaluations=result.n_evaluations,
        initial_diversity=result.initial_diversity,
        min_diversity=result.min_diversity,
        dev_bars=dev_bars,
        n_bars=int(panel.n_bars),
        wall_clock_seconds=result.wall_clock_seconds,
        history=list(result.history),
        config_snapshot=config.to_snapshot(),
    )
