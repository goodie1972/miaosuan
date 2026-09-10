"""多重检验校正（架构 §6.4，M12）。

搜索器在有限评估预算内比较**大量候选公式**（trials），「最高分」天然含选择偏差
（data snooping）。本模块提供两道校正：

* **Deflated Sharpe Ratio（DSR，Bailey & López de Prado）**：在「真实 Sharpe=0」零假设下，
  观测到的最优 Sharpe 需超过「N 次独立试验下的期望最大 Sharpe」才能判为显著。返回
  ``[0,1]`` 概率；越高越显著。
* **Bonferroni / Benjamini–Hochberg 阈值**：给定 ``alpha`` 与试验数 ``n_trials`` 给出
  校正后的显著性阈值（Bonferroni 为最保守的 fallback）。

口径约定：``observed_sharpe`` / ``sharpe_variance`` 必须**同一单位**（建议逐 bar Sharpe）；
``n_obs`` 为观测 bar 数。所有函数纯计算、无 IO、可复现。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.stats import norm

__all__ = [
    "BonferroniResult",
    "DSRResult",
    "benjamini_hochberg",
    "bonferroni_threshold",
    "deflated_sharpe_ratio",
]

#: 欧拉–马斯凯罗尼常数（DSR 期望最大 Sharpe 公式用）。
_EULER_GAMMA: float = 0.5772156649015329


@dataclass(frozen=True)
class BonferroniResult:
    """Bonferroni 校正结果。

    :param threshold: 校正后的显著性阈值（``alpha / n_trials``）。
    :param n_trials: 试验数。
    :param alpha: 名义显著性水平。
    """

    threshold: float
    n_trials: int
    alpha: float


@dataclass(frozen=True)
class DSRResult:
    """Deflated Sharpe Ratio 结果。

    :param dsr: 显著性概率（``[0,1]``，越高越显著）。
    :param sr0: 零假设下 N 次试验的期望最大 Sharpe。
    :param z: 标准化统计量。
    :param observed_sharpe: 观测 Sharpe。
    :param n_trials: 试验数。
    :param n_obs: 观测 bar 数。
    """

    dsr: float
    sr0: float
    z: float
    observed_sharpe: float
    n_trials: int
    n_obs: int

    def passes(self, threshold: float = 0.95) -> bool:
        """DSR 是否达到显著性阈值（默认 0.95）。"""
        return bool(self.dsr >= float(threshold))


def bonferroni_threshold(alpha: float, n_trials: int) -> BonferroniResult:
    """Bonferroni 阈值：``alpha / max(1, n_trials)``。

    :raises ValueError: ``alpha`` 不在 ``(0,1)`` 内。
    """
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("bonferroni_threshold: alpha 必须在 (0, 1) 内")
    n = max(1, int(n_trials))
    return BonferroniResult(threshold=float(alpha) / n, n_trials=n, alpha=float(alpha))


def deflated_sharpe_ratio(
    observed_sharpe: float,
    *,
    n_trials: int,
    n_obs: int,
    sharpe_variance: float | None = None,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> DSRResult:
    """计算 Deflated Sharpe Ratio（Bailey & López de Prado）。

    :param observed_sharpe: 观测 Sharpe（逐 bar 口径）。
    :param n_trials: 候选试验数（= 搜索期间评估过的公式数）。
    :param n_obs: 观测 bar 数（> 1）。
    :param sharpe_variance: 各试验 Sharpe 的方差 ``Var(SR)``；``None`` 时退化为
        ``1/n_obs``（SR 估计量自身方差的保守近似）——**这是无多试验 Sharpe 分布时的
        fallback，报告中会显式标注**。
    :param skew: PnL 偏度。
    :param kurtosis: PnL 峰度（正态 = 3）。

    :raises ValueError: ``n_obs < 2`` 或 ``sharpe_variance < 0``。
    """
    n = max(1, int(n_trials))
    obs = int(n_obs)
    if obs < 2:
        raise ValueError("deflated_sharpe_ratio: n_obs 必须 >= 2")
    var = (1.0 / obs) if sharpe_variance is None else float(sharpe_variance)
    if var < 0:
        raise ValueError("deflated_sharpe_ratio: sharpe_variance 不得为负")
    sigma = math.sqrt(var)

    # 零假设下 N 次独立试验的期望最大 Sharpe（SR 的极值理论近似）。
    z1 = float(norm.ppf(1.0 - 1.0 / n))
    z2 = float(norm.ppf(1.0 - 1.0 / (n * math.e)))
    sr0 = sigma * ((1.0 - _EULER_GAMMA) * z1 + _EULER_GAMMA * z2)

    sr = float(observed_sharpe)
    # Sharpe 估计量的方差（含偏度/峰度修正，Lo 2002 / Mertens 2002）。
    denom = max(obs - 1, 1)
    est_var = (1.0 - float(skew) * sr + (float(kurtosis) - 1.0) / 4.0 * sr * sr) / denom
    se = math.sqrt(max(est_var, 1e-18))
    z = (sr - sr0) / se
    dsr = float(norm.cdf(z))
    return DSRResult(
        dsr=dsr,
        sr0=float(sr0),
        z=float(z),
        observed_sharpe=sr,
        n_trials=n,
        n_obs=obs,
    )


def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> tuple[bool, ...]:
    """Benjamini–Hochberg FDR 控制：返回每个假设是否被拒（``True`` = 显著）。

    按 p 值升序，找最大 ``k`` 使得 ``p_(k) <= k/m * alpha``；前 ``k`` 个拒绝。

    :raises ValueError: ``alpha`` 不在 ``(0,1)`` 内或存在越界 p 值。
    """
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("benjamini_hochberg: alpha 必须在 (0, 1) 内")
    m = len(p_values)
    if m == 0:
        return ()
    for p in p_values:
        if not 0.0 <= float(p) <= 1.0:
            raise ValueError("benjamini_hochberg: p 值必须在 [0, 1] 内")
    order = sorted(range(m), key=lambda i: p_values[i])
    cutoff = -1
    for rank, idx in enumerate(order, start=1):
        if float(p_values[idx]) <= rank / m * float(alpha):
            cutoff = rank
    rejected = [False] * m
    for rank, idx in enumerate(order, start=1):
        if rank <= cutoff:
            rejected[idx] = True
    return tuple(rejected)
