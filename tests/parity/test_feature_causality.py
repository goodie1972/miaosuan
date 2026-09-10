"""M4 特征**因果性**断言（no look-ahead）。

因果性定义：``output[t]`` 仅依赖 ``input[<= t]``。据此有两个方向的等价断言：

1. **未来不污染过去（严格、无条件成立）** —— 扰动 ``input[t >= k]``，则
   ``output[t < k]`` 必须**逐位不变**。这是消除 look-ahead 泄露的核心保证，对全部
   65 个特征都必须成立（逐位相等，非容差）。
   （team-lead 要求的「改前 k 根、断言 t>k 不变」在窗口化特征上并不成立：``_norm``
   的 200 期窗口会让前 k 根的改动影响 ``t < k+199`` 的输出；其**对偶**命题
   ——「改未来、断言过去不变」才是我要的因果性，故按此实现）

2. **有界记忆：过去不影响足够远的未来（在记忆上界之外）** —— 扰动 ``input[:k]``，
   则 ``output[t >= k + M]`` 必须逐位不变，其中 ``M`` 为全部因果窗口的最大回看长度
   （``_MAX_LOOKBACK = 600``，覆盖 robust_norm 200 + trix 三重 EMA ~312 等）与
   ``long`` 用例 T=800。**例外**（记忆无界，仅参与方向 1）：

   * ``SUPERTREND_DIR``：带状态方向标志（一旦翻转持续保持）；
   * ``OBV_SLOPE`` / ``AD_LINE_SLOPE``：基于 ``cumsum``，早期改动量会平移整条累加线，
     经 ``_linear_slope`` 的 ``slope/xm`` 归一化后影响所有后续 ``t``（非 look-ahead，
     方向 1 仍严格成立）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import feature_cases  # noqa: E402

from miaosuan.core.features import FEATURE_NAMES, compute_features  # noqa: E402

pytestmark = pytest.mark.parity

_FIELDS = feature_cases.FIELDS

#: 方向 2 的最大回看长度（覆盖 robust_norm 200 + trix 三重 EMA ~312 等）
_MAX_LOOKBACK = 600

#: 方向 2 中记忆无界的特征（带状态递推 / cumsum 平移归一化），不参与方向 2
_UNBOUNDED_STATE = {"SUPERTREND_DIR", "OBV_SLOPE", "AD_LINE_SLOPE"}

#: 方向 1：base 用例，扰动后 k 根
_K_A = 160

#: 方向 2：long 用例，扰动前 k 根
_K_B = 80


def _perturb(payload: dict, start: int, end: int, factor: float = 1.37) -> dict:
    """把 ``payload`` 各字段的 ``[:, start:end]`` 乘 ``factor``，返回新的 raw_dict。"""
    raw: dict[str, np.ndarray] = {}
    for f in _FIELDS:
        arr = payload[f].astype(np.float32).copy()
        arr[:, start:end] = arr[:, start:end] * factor
        raw[f] = arr
    return raw


@pytest.fixture(scope="session")
def causality() -> dict:
    cases = feature_cases.build_cases()
    base = cases["base"]
    long = cases["long"]
    t_base = base["close"].shape[1]
    t_long = long["close"].shape[1]

    base_orig = compute_features({f: base[f] for f in _FIELDS})
    base_pert = compute_features(_perturb(base, _K_A, t_base))

    long_orig = compute_features({f: long[f] for f in _FIELDS})
    long_pert = compute_features(_perturb(long, 0, _K_B))

    return {
        "base_orig": base_orig,
        "base_pert_suffix": base_pert,
        "long_orig": long_orig,
        "long_pert_prefix": long_pert,
        "k_a": _K_A,
        "k_b": _K_B,
        "t_long": t_long,
    }


def test_perturbation_is_effective(causality: dict) -> None:
    """哨兵：确认扰动确实改变了应改变的区域（否则因果断言会『真空通过』）。"""
    orig = causality["base_orig"]
    pert = causality["base_pert_suffix"]
    k = causality["k_a"]
    assert not np.array_equal(orig[:, :, k:], pert[:, :, k:]), (
        "扰动未改变任何未来输出，因果性测试失去意义"
    )


@pytest.mark.parametrize("feat_idx", range(65), ids=list(FEATURE_NAMES))
def test_no_lookahead_future_does_not_affect_past(feat_idx: int, causality: dict) -> None:
    """方向 1（严格）：扰动未来（t>=k），过去（t<k）必须逐位不变。"""
    k = causality["k_a"]
    orig = causality["base_orig"][:, feat_idx, :k]
    pert = causality["base_pert_suffix"][:, feat_idx, :k]
    assert np.array_equal(orig, pert, equal_nan=True), (
        f"{FEATURE_NAMES[feat_idx]} 存在 look-ahead：改动 t>={k} 影响了 t<{k} 的输出"
    )


@pytest.mark.parametrize("feat_idx", range(65), ids=list(FEATURE_NAMES))
def test_bounded_memory_past_does_not_affect_far_future(feat_idx: int, causality: dict) -> None:
    """方向 2（有界记忆）：扰动前 k 根，``t >= k + M`` 必须逐位不变。"""
    name = FEATURE_NAMES[feat_idx]
    if name in _UNBOUNDED_STATE:
        pytest.skip(f"{name} 记忆无界（带状态 / cumsum 平移），不参与方向 2")
    k = causality["k_b"]
    start = k + _MAX_LOOKBACK
    orig = causality["long_orig"][:, feat_idx, start:]
    pert = causality["long_pert_prefix"][:, feat_idx, start:]
    assert np.array_equal(orig, pert, equal_nan=True), (
        f"{name} 回看超出 {_MAX_LOOKBACK}：改动 t<{k} 影响了 t>={start} 的输出"
    )
