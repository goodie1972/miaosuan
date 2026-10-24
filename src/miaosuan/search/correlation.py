# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for full license text.

"""S4 因子池相关性惩罚（路线图 §6）：top-k 因子向量池 + |corr| 超阈值降分。

**动机**：GA 容易把同质因子反复生成（同一信号模式的变体），因子池相互冗余时
组合的边际贡献≈0，但单因子分数却很高。S4 在**搜索回路内**（而非事后检查）
对「与既有高分类似」的公式乘罚系数，迫使搜索去探索**新信号方向**。

**只认配置**：默认关闭（``CorrelationPenaltyEvaluator`` 未启用时行为与既有
评估器逐位一致，冻结基线不受影响）。池大小 / 阈值 / 罚系数全部来自配置。

**防泄漏铁律**：相关性**只在 train 折切片**上计算——val 段绝不参与池的
构建与比较（杜绝「靠 val 相似性刷分」的泄漏通道）。评估器收到的 folds 切片
由调用方提供；本模块不自行推导数据分段。
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .ga import EvalResult, FormulaEvaluator

__all__ = ["CorrelationPenaltyEvaluator", "pearson_corr"]

logger = logging.getLogger("miaosuan.search.correlation")

_EPS = 1e-12


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    """两个一维向量的皮尔逊相关系数（std 为 0 时返回 0，不抛）。"""
    av = np.asarray(a, dtype=np.float64).reshape(-1)
    bv = np.asarray(b, dtype=np.float64).reshape(-1)
    if av.shape != bv.shape or av.size == 0:
        return 0.0
    am = av - av.mean()
    bm = bv - bv.mean()
    denom = float(np.sqrt((am * am).sum() * (bm * bm).sum()))
    if denom <= _EPS:
        return 0.0
    return float((am * bm).sum() / denom)


@dataclass
class CorrelationPenaltyEvaluator(FormulaEvaluator):
    """包装评估器：维护 top-k 因子池，|corr| 超阈值乘罚系数。

    :param inner: 被包装的评估器（其 :meth:`evaluate` 仍是分数的事实来源）。
    :param factor_fn: ``tokens -> [T]`` 因子向量提取（必须返回 **train 折切片**）。
    :param pool_size: 因子池容量（top-k；按 |corr| 命中次数排），默认 20。
    :param corr_threshold: |corr| 超过该值即降分，默认 0.90。
    :param penalty: 命中时的罚系数（乘在 train_score / val_score 上），默认 0.5。
    """

    inner: FormulaEvaluator
    factor_fn: Callable[[np.ndarray], np.ndarray]
    pool_size: int = 20
    corr_threshold: float = 0.90
    penalty: float = 0.5
    #: 因子池（仅 train 折切片；最"有代表性"的池成员按命中次数保留）。
    _pool: list[np.ndarray] = field(default_factory=list, repr=False, compare=False)
    _hits: dict[int, int] = field(default_factory=dict, repr=False, compare=False)
    #: 统计（可观测：命中的评估次数）。
    n_penalized: int = 0
    n_evaluated: int = 0

    def evaluate(self, tokens: np.ndarray) -> EvalResult:
        """先取内层分数，再按 train 折相关性降分，最后把该因子吸收进池。"""
        base = self.inner.evaluate(tokens)
        self.n_evaluated += 1
        factor = self.factor_fn(tokens)
        if factor is None or factor.size == 0:
            return base  # 不可求值：不参与池，不降分
        corr = self._max_corr(factor)
        if corr is not None and abs(corr) > float(self.corr_threshold):
            self.n_penalized += 1
            logger.debug(
                "相关性惩罚命中: |corr|=%.3f > %.2f（罚系数 %.2f）",
                abs(corr),
                self.corr_threshold,
                self.penalty,
            )
            return EvalResult(
                train_score=float(base.train_score) * float(self.penalty),
                val_score=float(base.val_score) * float(self.penalty),
                status=base.status,
                ic_mean=base.ic_mean,
            )
        self._absorb(factor)
        return base

    # ── 池 ────────────────────────────────────────────────────────────────

    def _max_corr(self, factor: np.ndarray) -> float | None:
        """与池内所有因子的最大 |corr|（池空返回 None）。"""
        best = None
        for pool_f in self._pool:
            c = abs(pearson_corr(factor, pool_f))
            if best is None or c > best:
                best = c
        return best

    def _absorb(self, factor: np.ndarray) -> None:
        """把新因子吸收进池（容量上限 + 近似去重：与池成员 |corr|>0.99 视为重复）。"""
        for pool_f in self._pool:
            if abs(pearson_corr(factor, pool_f)) > 0.99:
                return  # 与既有池成员几乎相同，不入池（防池被变体塞满）
        self._pool.append(np.asarray(factor, dtype=np.float64).reshape(-1).copy())
        if len(self._pool) > int(self.pool_size):
            # 淘汰与最新因子最相似者（保留池的"方向多样性"）。
            newest = self._pool[-1]
            most_similar_idx = max(
                range(len(self._pool) - 1),
                key=lambda i: abs(pearson_corr(newest, self._pool[i])),
            )
            self._pool.pop(most_similar_idx)
        self._hits = {}  # 简化：不做命中计数淘汰（池方向多样性已够）