# -*- coding: utf-8 -*-
# Copyright (c) 2026 Original Author
# Copyright (c) 2026 Your Company
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""RPN 遗传搜索引擎（架构 §1.3、§6.2，M10–M11）。

本模块把「表示/算子」（:mod:`miaosuan.search.rpn`）、「预算/早停」
（:mod:`miaosuan.search.budget`）、「岛屿模型」（:mod:`miaosuan.search.islands`）与
「AM 口径适应度评估器」（:class:`AMFitnessEvaluator`）组装为可复现的进化引擎
:class:`RpnGA`。

设计要点（与团队任务书 M10/M11 对齐）：

* **表示**：定长 token 数组 ``L=8``，生成/交叉/变异全部经栈深可行性修复 → **不存在不可求值个体**；
  ``min_hamming`` 为与「精英库 + 本代已接受个体」的汉明距离**硬下界**（三档一致 = 2，治 R2 熵坍塌）。
* **适应度 = AM 口径**：逐折 ``MT5Backtest.evaluate_fold`` → IC 门控（``IC>0.01×1.15`` / ``IC<−0.01×0.75``）
  → ``val_score`` 取各折验证分均值，再减重复惩罚；``train_score`` 同法。**与冻结 AlphaMaster 的
  ``engine._eval_formula_task`` 逐点对齐**（``REWARD_ALPHA=1.0``、``WF_GAP=20``、``n_folds=5``）。
* **预算 = 三档**：种群 / 精英 / 锦标赛 / patience / 移民 / 岛屿参数**全部来自** :class:`Budget`
  （即 §6.2 分档表），算子概率来自 :class:`~miaosuan.config.GASearchConfig`。**墙钟为硬约束**。
* **年化口径**：``periods_per_year`` 由 :class:`~miaosuan.market.profiles.FrozenMarketProfile`.bars_per_year
  注入 —— 本模块**不硬编码 6240**。
* **岛屿迁移 = 纯数据搬运**：迁移的个体已评估，**绝不重复评估**（评估计数仅统计缓存未命中）。

依赖方向：仅依赖 ``core`` / ``config`` 与同包 ``search`` 子模块；不 import ``gate`` / ``adapters`` / ``cli``。
"""

from __future__ import annotations

import copy
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from ..config import GASearchConfig
from ..core.backtest import MT5Backtest
from ..core.vm import StackVM
from .budget import (
    STOP_RUNNING,
    Budget,
)
from .islands import IslandModel, partition_population
from .rpn import (
    FORMULA_LEN,
    Individual,
    crossover,
    mutate,
    random_feasible,
)

__all__ = [
    "AMFitnessEvaluator",
    "EvalResult",
    "FormulaEvaluator",
    "GAStats",
    "GAResult",
    "RpnGA",
    "build_walk_forward_folds",
]

# ── AM 口径常数（与冻结 AlphaMaster ``model_core/config.py`` 对齐；硬约束，不得擅改）──
IC_GATE_THRESH: float = 0.01
IC_GATE_MULT: float = 1.15
IC_NEG_MULT: float = 0.75
REWARD_ALPHA: float = 1.0
WF_GAP: int = 20
DEFAULT_N_FOLDS: int = 5

#: 不可求值 / 退化常数公式的惩罚分（与 AM ``_eval_formula_task`` 一致）。
NONE_SCORE: float = -5.0
CONST_SCORE: float = -2.0

#: 因子有效性下限（``std < 1e-4`` 视为常数，AM 一致）。
_CONST_STD: float = 1e-4


# ── 适应度评估接口（依赖倒置：搜索引擎只认协议，不绑定具体实现）───────────────


@dataclass(frozen=True)
class EvalResult:
    """单条公式的评估结果（AM 口径）。

    :param train_score: 开发集训练分（各折 ``train_score`` 的 IC 门控均值 − 重复惩罚）。
    :param val_score: 开发集验证分（各折 ``val_score`` 的 IC 门控均值 − 重复惩罚）。
    :param status: ``"ok"`` / ``"none"`` / ``"const"`` / ``"error"``。
    :param ic_mean: 各折训练 IC 的均值（诊断用）。
    """

    train_score: float
    val_score: float
    status: str = "ok"
    ic_mean: float = 0.0


@runtime_checkable
class FormulaEvaluator(Protocol):
    """公式适应度评估器协议（可注入，便于以廉价合成评估器做单测）。"""

    def evaluate(self, tokens: np.ndarray) -> EvalResult:
        """评估一条定长 RPN 公式，返回 :class:`EvalResult`。"""
        ...


# ── AM 口径辅助函数（逐点对齐冻结 AlphaMaster）──────────────────────────────


def build_walk_forward_folds(T: int, n_folds: int = DEFAULT_N_FOLDS, gap: int = WF_GAP) -> list[dict[str, int]]:
    """构建 Walk-Forward 折叠（AM ``_build_walk_forward_folds`` 逐行等价）。

    rolling window：第 ``k`` 折训练 ``[(k−1)·fold_size, k·fold_size)``，验证
    ``[k·fold_size + gap, k·fold_size + gap + fold_size)``（末端以 ``min(·, T)`` 收敛）。
    ``fold_size < 2`` 时退化为单折全量评估。
    """
    fold_size = T // int(n_folds)
    if fold_size < 2:
        return [{"train_start": 0, "train_end": T, "val_start": 0, "val_end": T, "gap": 0}]
    total_required = fold_size * int(n_folds) + int(gap) * (int(n_folds) - 1)
    use_gap = int(gap)
    if total_required > T:
        use_gap = max(0, (T - fold_size * int(n_folds)) // int(n_folds))
    folds: list[dict[str, int]] = []
    for k in range(1, int(n_folds)):
        train_start = (k - 1) * fold_size
        train_end = k * fold_size
        val_start = train_end + use_gap
        val_end = min(val_start + fold_size, T)
        if val_start >= T or val_end <= val_start:
            break
        folds.append(
            {
                "train_start": train_start,
                "train_end": train_end,
                "val_start": val_start,
                "val_end": val_end,
                "gap": use_gap,
            }
        )
    if not folds:
        return [{"train_start": 0, "train_end": T, "val_start": 0, "val_end": T, "gap": 0}]
    return folds


def _compute_ic(factor: np.ndarray, target_ret: np.ndarray) -> tuple[float, float]:
    """时序 IC（每样本内部 ``factor[t]`` vs ``ret[t+1]``）的均值与稳定性。

    与 AM ``AlphaEngine._compute_ic`` 等价：``std`` 用总体口径（``ddof=0``）。
    """
    if factor.ndim != 2:
        raise ValueError("_compute_ic: factor 必须为 [N, T]")
    n, t = factor.shape
    if t < 2:
        return 0.0, 0.0
    ics: list[float] = []
    for i in range(n):
        x = factor[i, :-1]
        y = target_ret[i, 1:]
        xm = x - x.mean()
        ym = y - y.mean()
        sx = float(np.sqrt((xm**2).mean()))
        sy = float(np.sqrt((ym**2).mean()))
        if sx < 1e-6 or sy < 1e-6:
            continue
        ics.append(float((xm * ym).mean() / (sx * sy + 1e-8)))
    if not ics:
        return 0.0, 0.0
    arr = np.asarray(ics, dtype=np.float64)
    ic_mean = float(arr.mean())
    ic_stab = float(ic_mean / (arr.std() + 1e-6)) if arr.size >= 2 else 0.0
    return ic_mean, ic_stab


def _apply_ic_gate(reward: float, ic_mean: float) -> float:
    """IC 门控（AM ``_apply_ic_gate`` 等价，方向敏感、量纲无关）。"""
    if ic_mean > IC_GATE_THRESH:
        return reward * IC_GATE_MULT
    if ic_mean < -IC_GATE_THRESH:
        return reward * IC_NEG_MULT
    return reward


def _repetition_penalty(tokens: np.ndarray) -> float:
    """重复惩罚（AM ``_repetition_penalty`` 等价）：相邻相同 token 累计每满 2 次罚 0.3。"""
    penalty = 0.0
    count = 1
    toks = [int(t) for t in tokens]
    for i in range(1, len(toks)):
        if toks[i] == toks[i - 1]:
            count += 1
            if count >= 2:
                penalty += 0.3
        else:
            count = 1
    return float(penalty)


class AMFitnessEvaluator:
    """AM 口径适应度评估器（5 折滚动 Walk-Forward + IC 门控 + 重复惩罚）。

    :param features: ``[N, F, T]`` 特征张量（由 :func:`miaosuan.core.features.compute_features` 产出）。
    :param target_ret: ``[N, T]`` 前瞻收益（``target_ret[t] = log(open[t+2]/open[t+1])``）。
    :param cost_rate: 单边成本率（来自 :class:`~miaosuan.market.profiles.FrozenMarketProfile`.cost_model）。
    :param periods_per_year: 年化因子（来自 profile ``bars_per_year``；**不硬编码**）。
    :param n_folds: 滚动折数（默认 5，与 AM 一致）。
    :param gap: 折间间隔（默认 20，与 AM ``WF_GAP`` 一致）。
    :param reward_mode: 回测奖励模式（默认 ``"ftmo"``，与 AM 一致）。
    :param reward_alpha: 训练分缩放（默认 1.0，与 AM 一致）。
    :param folds: 可选显式折（否则按 ``T/n_folds`` 自动构建）。
    """

    def __init__(
        self,
        features: np.ndarray,
        target_ret: np.ndarray,
        *,
        cost_rate: float,
        periods_per_year: int,
        n_folds: int = DEFAULT_N_FOLDS,
        gap: int = WF_GAP,
        reward_mode: str = "ftmo",
        reward_alpha: float = REWARD_ALPHA,
        folds: list[dict[str, int]] | None = None,
        vm: StackVM | None = None,
    ) -> None:
        self.features = np.asarray(features)
        self.target_ret = np.asarray(target_ret)
        if self.features.shape[0] != self.target_ret.shape[0]:
            raise ValueError("features 与 target_ret 的 N 维不一致")
        self.cost_rate = float(cost_rate)
        self.periods_per_year = int(periods_per_year)
        self.n_folds = int(n_folds)
        self.gap = int(gap)
        self.reward_alpha = float(reward_alpha)
        self.reward_mode = str(reward_mode)
        self.vm = vm if vm is not None else StackVM()
        self.bt = MT5Backtest(
            cost_rate=self.cost_rate,
            periods_per_year=self.periods_per_year,
            reward_mode=self.reward_mode,
        )
        t = self.target_ret.shape[1]
        self.folds: list[dict[str, int]] = (
            list(folds) if folds is not None else build_walk_forward_folds(t, self.n_folds, self.gap)
        )
        self.n_evaluated = 0

    # ── 折分（供门禁复用）──────────────────────────────────────────────────

    def fold_scores(self, tokens: np.ndarray) -> tuple[list[float], list[float]]:
        """返回逐折 ``(train_scores, val_scores)``（IC 门控后，未减重复惩罚）。"""
        res = self.vm.execute([int(t) for t in tokens], self.features)
        if res is None:
            return [], []
        train_scores: list[float] = []
        val_scores: list[float] = []
        for f in self.folds:
            tr_sc, vl_sc = self.bt.evaluate_fold(
                res,
                self.target_ret,
                f["train_start"],
                f["train_end"],
                f["val_start"],
                f["val_end"],
            )
            ic_m, _ = _compute_ic(
                res[:, f["train_start"] : f["train_end"]],
                self.target_ret[:, f["train_start"] : f["train_end"]],
            )
            train_scores.append(self.reward_alpha * _apply_ic_gate(float(tr_sc), ic_m))
            ic_v, _ = _compute_ic(
                res[:, f["val_start"] : f["val_end"]],
                self.target_ret[:, f["val_start"] : f["val_end"]],
            )
            val_scores.append(_apply_ic_gate(float(vl_sc), ic_v))
        return train_scores, val_scores

    # ── 协议实现 ───────────────────────────────────────────────────────────

    def evaluate(self, tokens: np.ndarray) -> EvalResult:
        """评估一条公式，返回 AM 口径的 :class:`EvalResult`。"""
        self.n_evaluated += 1
        res = self.vm.execute([int(t) for t in tokens], self.features)
        if res is None:
            return EvalResult(NONE_SCORE, NONE_SCORE, "none")
        if float(res.std()) < _CONST_STD:
            return EvalResult(CONST_SCORE, CONST_SCORE, "const")

        train_scores, val_scores = self.fold_scores(tokens)
        if not val_scores:
            return EvalResult(NONE_SCORE, NONE_SCORE, "none")
        train_score = float(np.mean(train_scores))
        val_score = float(np.mean(val_scores))

        # 重复惩罚（AM 一致：train/val 同时扣）。
        penalty = _repetition_penalty(tokens)
        train_score -= penalty
        val_score -= penalty

        ics: list[float] = []
        for f in self.folds:
            ic_m, _ = _compute_ic(
                res[:, f["train_start"] : f["train_end"]],
                self.target_ret[:, f["train_start"] : f["train_end"]],
            )
            ics.append(ic_m)
        ic_mean = float(np.mean(ics)) if ics else 0.0
        return EvalResult(train_score, val_score, "ok", ic_mean)


# ── 统计与结果 ─────────────────────────────────────────────────────────────


@dataclass
class GAStats:
    """单代统计快照（可机器验证的多样性/评估证据）。

    :param generation: 已完成的代数（0 基）。
    :param best_fitness: 当前跨岛最优适配度。
    :param mean_fitness: 当前跨岛平均适配度。
    :param diversity: 当前平均两两汉明距离（跨全部岛）。
    :param population_size: 当前总种群规模。
    :param n_evaluations: 累计真实评估次数（缓存未命中）。
    :param n_immigrants: 累计注入的移民数。
    :param n_diversity_rejects: 累计因 ``min_hamming`` 被拒的候选数。
    """

    generation: int
    best_fitness: float
    mean_fitness: float
    diversity: float
    population_size: int
    n_evaluations: int
    n_immigrants: int
    n_diversity_rejects: int


@dataclass
class GAResult:
    """一次 :meth:`RpnGA.run` 的完整结果（含可审计证据）。"""

    best: Individual
    history: list[GAStats]
    stop_reason: str
    generations: int
    n_evaluations: int
    initial_diversity: float
    min_diversity: float
    wall_clock_seconds: float
    n_diversity_immigrants: int = 0
    diversity_floor: float = 0.0
    populations: list[list[Individual]] = field(default_factory=list)

    @property
    def diversity_ratio(self) -> float:
        """全程最低多样性 / 初始多样性（验收 #3：≥ 0.4）。"""
        if self.initial_diversity <= 0:
            return 1.0
        return self.min_diversity / self.initial_diversity


# ── 进化引擎 ───────────────────────────────────────────────────────────────


class RpnGA:
    """定长 RPN 遗传搜索引擎（岛屿 + 多样性硬约束 + 移民 + 三档预算）。

    :param config: 搜索器配置（算子概率 / 公式长度）。
    :param seed: 随机种子（完全可复现；内部仅用 ``np.random.Generator``）。
    :param length: 公式长度（默认 :data:`~miaosuan.search.rpn.FORMULA_LEN` = 8）。
    :param diversity_floor_ratio: 多样性地板（占初始平均两两汉明距离的比例）。
        当某一代种群多样性跌破 ``diversity_floor_ratio × 初始多样性`` 时，注入**最大多样性**
        随机个体顶替随机成员，把多样性拉回地板之上 —— 保证「全程多样性 ≥ 初始 40%」这一
        无熵坍塌验收（默认 0.45，留 5pp 余量；设 0 关闭）。补充（非替代）``min_hamming`` 硬约束。
    """

    def __init__(
        self,
        config: GASearchConfig,
        *,
        seed: int,
        length: int = FORMULA_LEN,
        diversity_floor_ratio: float = 0.45,
    ) -> None:
        self.config = config
        self.seed = int(seed)
        self.length = int(length)
        self.diversity_floor_ratio = float(diversity_floor_ratio)
        self.rng = np.random.default_rng(self.seed)
        # 评估缓存（token 元组 → EvalResult），保证同一公式**只评估一次**。
        self._cache: dict[tuple[int, ...], EvalResult] = {}
        self.n_evaluations = 0
        self.n_immigrants = 0
        self.n_diversity_immigrants = 0
        self.n_diversity_rejects = 0
        self.history: list[GAStats] = []
        self.final_populations: list[list[Individual]] = []

    # ── 基础构件 ────────────────────────────────────────────────────────────

    def _random_individual(self, generation: int) -> Individual:
        toks = random_feasible(self.rng, self.length)
        return Individual(tokens=toks, birth_gen=generation)

    def _evaluate(self, ind: Individual, evaluator: FormulaEvaluator) -> None:
        """评估个体（带缓存；缓存命中不计入真实评估次数）。"""
        key = ind.key()
        cached = self._cache.get(key)
        if cached is None:
            cached = evaluator.evaluate(ind.tokens)
            self._cache[key] = cached
            self.n_evaluations += 1
        ind.train_score = cached.train_score
        ind.val_score = cached.val_score
        ind.fitness = cached.val_score
        ind.status = cached.status

    def _init_population(self, pop_size: int, evaluator: FormulaEvaluator) -> list[Individual]:
        """随机可行初始化（去重 + 全部评估）。"""
        pop: list[Individual] = []
        seen: set[tuple[int, ...]] = set()
        tries = 0
        max_tries = int(pop_size) * 100
        while len(pop) < int(pop_size) and tries < max_tries:
            tries += 1
            ind = self._random_individual(0)
            if ind.key() in seen:
                continue
            seen.add(ind.key())
            self._evaluate(ind, evaluator)
            pop.append(ind)
        while len(pop) < int(pop_size):  # pragma: no cover - 极端兜底（去重空间枯竭）
            ind = self._random_individual(0)
            self._evaluate(ind, evaluator)
            pop.append(ind)
        return pop

    def _tournament(self, pool: list[Individual], k: int) -> Individual:
        """k 元锦标赛选择：随机取 k 个，返回适配度最高者。"""
        kk = max(1, min(int(k), len(pool)))
        idx = self.rng.integers(0, len(pool), size=kk)
        best = pool[int(idx[0])]
        for j in idx[1:]:
            cand = pool[int(j)]
            if cand.fitness > best.fitness:
                best = cand
        return best

    @staticmethod
    def _accept_diversity(tokens: np.ndarray, archive: np.ndarray, min_hamming: int) -> bool:
        """新个体与「精英库 + 本代已接受个体」（``archive`` 逐行）的最小汉明距离 ≥ ``min_hamming``。"""
        if int(min_hamming) <= 0 or archive.shape[0] == 0:
            return True
        dist = np.count_nonzero(archive != tokens[None, :], axis=1)
        return bool(int(dist.min()) >= int(min_hamming))

    # ── 单岛演化 ────────────────────────────────────────────────────────────

    def _evolve_island(
        self,
        island: list[Individual],
        evaluator: FormulaEvaluator,
        budget: Budget,
        generation: int,
        size: int,
    ) -> list[Individual]:
        """演化单个岛：精英保留 + 锦标赛 + 交叉/变异 + 多样性硬约束 + 移民。"""
        ranked = sorted(island, key=lambda x: x.fitness, reverse=True)
        elites = [copy.deepcopy(ind) for ind in ranked[: int(budget.elite_size)]]
        next_island: list[Individual] = list(elites)
        archive = (
            np.asarray([e.tokens for e in elites], dtype=np.int64)
            if elites
            else np.empty((0, self.length), dtype=np.int64)
        )

        # 移民判定：连续无改进达到阈值 → 本代注入随机移民（比例取自 Budget）。
        inject = budget.generations_since_improvement(generation) >= int(budget.immigrant_trigger)
        n_immigrants = int(round(size * float(budget.immigrant_ratio))) if inject else 0
        n_immigrants = min(n_immigrants, max(0, size - len(next_island)))
        offspring_target = size - len(next_island) - n_immigrants

        # ── 后代（交叉 + 变异 + 多样性硬约束）─────────────────────────────
        attempts = 0
        max_attempts = max(64, size * 24)
        while len(next_island) < int(budget.elite_size) + offspring_target and attempts < max_attempts:
            attempts += 1
            parent_a = self._tournament(island, budget.tournament_k)
            if self.rng.random() < float(self.config.p_crossover) and len(island) >= 2:
                parent_b = self._tournament(island, budget.tournament_k)
                child_tokens = crossover(parent_a.tokens, parent_b.tokens, self.rng, length=self.length)
            else:
                child_tokens = parent_a.tokens
            child_tokens = mutate(
                child_tokens,
                self.rng,
                p_point=float(self.config.p_mutate_point),
                p_reset=float(self.config.p_mutate_subtree),
            )
            if not self._accept_diversity(child_tokens, archive, budget.min_hamming):
                self.n_diversity_rejects += 1
                continue
            child = Individual(tokens=child_tokens, birth_gen=generation)
            self._evaluate(child, evaluator)
            next_island.append(child)
            archive = np.vstack([archive, child_tokens[None, :]])

        # ── 移民（随机可行 + 多样性硬约束）────────────────────────────────
        imm_added = 0
        imm_attempts = 0
        max_imm_attempts = max(64, n_immigrants * 60)
        while imm_added < n_immigrants and imm_attempts < max_imm_attempts:
            imm_attempts += 1
            imm_tokens = random_feasible(self.rng, self.length)
            if not self._accept_diversity(imm_tokens, archive, budget.min_hamming):
                self.n_diversity_rejects += 1
                continue
            immigrant = Individual(tokens=imm_tokens, birth_gen=generation)
            self._evaluate(immigrant, evaluator)
            next_island.append(immigrant)
            archive = np.vstack([archive, imm_tokens[None, :]])
            imm_added += 1
            self.n_immigrants += 1

        # ── 兜底补齐（极端情况下不缩种群）─────────────────────────────────
        while len(next_island) < int(size):  # pragma: no cover - 有界尝试后兜底
            ind = self._random_individual(generation)
            self._evaluate(ind, evaluator)
            next_island.append(ind)

        return next_island[: int(size)]

    # ── 主循环 ──────────────────────────────────────────────────────────────

    def run(
        self,
        budget: Budget,
        evaluator: FormulaEvaluator,
        *,
        on_generation: Callable[[GAStats], None] | None = None,
    ) -> Individual:
        """执行一次进化搜索，返回跨岛最优个体。

        :param budget: 三档预算（提供种群/选择/移民/岛屿/早停参数，且被就地更新停止原因）。
        :param evaluator: 适应度评估器（可实现 :class:`FormulaEvaluator`）。
        :param on_generation: 可选逐代回调（用于日志/进度）。
        """
        self._cache.clear()
        self.n_evaluations = 0
        self.n_immigrants = 0
        self.n_diversity_immigrants = 0
        self.n_diversity_rejects = 0
        self.history = []
        budget.reset()

        island_model = IslandModel(
            n_islands=int(budget.island_count),
            migrate_every=max(1, int(budget.island_migrate_every)),
            migrate_top=max(1, int(budget.island_migrate_top)),
        )
        population = self._init_population(int(budget.pop_size), evaluator)
        populations = partition_population(population, int(budget.island_count))
        island_sizes = [len(p) for p in populations]

        start = time.perf_counter()
        initial_diversity = island_model.diversity(populations)
        diversity_floor = max(0.0, self.diversity_floor_ratio) * initial_diversity
        min_diversity = initial_diversity
        best = copy.deepcopy(island_model.best(populations))

        generation = 0
        stop_reason = STOP_RUNNING
        while True:
            elapsed = time.perf_counter() - start
            diversity = island_model.diversity(populations)
            min_diversity = min(min_diversity, diversity)
            stats = GAStats(
                generation=generation,
                best_fitness=float(best.fitness),
                mean_fitness=float(
                    np.mean([ind.fitness for ind in island_model.all_individuals(populations)])
                ),
                diversity=float(diversity),
                population_size=int(sum(island_sizes)),
                n_evaluations=self.n_evaluations,
                n_immigrants=self.n_immigrants,
                n_diversity_rejects=self.n_diversity_rejects,
            )
            self.history.append(stats)
            if on_generation is not None:
                on_generation(stats)

            if budget.should_stop(generation, float(best.fitness), elapsed, diversity=diversity):
                stop_reason = budget.stop_reason()
                break

            if island_model.should_migrate(generation):
                island_model.migrate(populations)
                migrated_best = island_model.best(populations)
                if migrated_best.fitness > best.fitness:
                    best = copy.deepcopy(migrated_best)

            populations = [
                self._evolve_island(island, evaluator, budget, generation, size)
                for island, size in zip(populations, island_sizes, strict=True)
            ]
            # 多样性地板：跌破则注入最大多样性随机个体，防熵坍塌（验收 #3 兜底）。
            if diversity_floor > 0.0:
                self._restore_diversity(populations, island_model, evaluator, diversity_floor)
            generation += 1
            current_best = island_model.best(populations)
            if current_best.fitness > best.fitness:
                best = copy.deepcopy(current_best)

        self.final_populations = populations
        self.last_result = GAResult(
            best=best,
            history=list(self.history),
            stop_reason=stop_reason,
            generations=generation,
            n_evaluations=self.n_evaluations,
            initial_diversity=float(initial_diversity),
            min_diversity=float(min_diversity),
            wall_clock_seconds=float(time.perf_counter() - start),
            n_diversity_immigrants=self.n_diversity_immigrants,
            diversity_floor=float(diversity_floor),
            populations=populations,
        )
        return best

    def _restore_diversity(
        self,
        populations: list[list[Individual]],
        island_model: IslandModel,
        evaluator: FormulaEvaluator,
        floor: float,
        *,
        max_fraction: float = 0.25,
    ) -> None:
        """把种群多样性拉回地板上（原地修改）。

        每次注入一个**最大多样性**随机可行个体（新评估），顶替随机岛屿的随机成员；批量注入
        后重新测量，直至达到地板或触及上限（``max_fraction × 总种群``，防止单代注入过多）。
        """
        total = sum(len(p) for p in populations)
        if total == 0:  # pragma: no cover - 空种群不应发生
            return
        cap = max(1, int(max_fraction * total))
        step = max(1, cap // 4)
        injected = 0
        while injected < cap and island_model.diversity(populations) < floor:
            for _ in range(step):
                if injected >= cap:
                    break
                sizes = [len(p) for p in populations]
                target = int(np.argmax(sizes))
                pop = populations[target]
                if not pop:  # pragma: no cover - 空岛不应发生
                    continue
                idx = int(self.rng.integers(0, len(pop)))
                immigrant = Individual(
                    tokens=random_feasible(self.rng, self.length), birth_gen=-1
                )
                self._evaluate(immigrant, evaluator)
                pop[idx] = immigrant
                injected += 1
                self.n_diversity_immigrants += 1

    def top_individuals(self, k: int = 5) -> list[Individual]:
        """返回最终种群中适配度最高的 ``k`` 个**去重**个体（用于候选汇总/门禁）。"""
        pool = [ind for pop in self.final_populations for ind in pop]
        best_by_key: dict[tuple[int, ...], Individual] = {}
        for ind in pool:
            key = ind.key()
            if key not in best_by_key or ind.fitness > best_by_key[key].fitness:
                best_by_key[key] = ind
        ranked = sorted(best_by_key.values(), key=lambda x: x.fitness, reverse=True)
        return ranked[: int(k)]

    #: 最近一次运行的结果（``run`` 后可用）。
    last_result: GAResult | None = None
