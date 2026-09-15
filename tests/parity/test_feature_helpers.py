# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""M4 底层 helper 对拍（**紧容差** 1e-5）：隔离「归一化放大」，证明移植正确。

65 特征的输出统一经过 ``_norm``（median/MAD 稳健归一化），会把 1e-7 量级的 float32
归约顺序差异放大 ``1/MAD`` 倍。为把「算法正确性」与「归一化放大」解耦，本文件用
**同一份输入**（各 case 的 close/high/low）分别调用：

* 妙算 numpy helper（``miaosuan.core.features`` 的模块级 helper）；
* AM torch helper（基准 ``features_baseline.npz`` 的 ``helper_{key}_{case}``）；

以 ``atol=rtol=1e-5`` 的**紧容差**比对（实测最坏约 3e-2，含 300+ 倍余量）。这样：
helper 逐一对齐 → 特征由 helper 组合而成 → 输出层差异纯属归一化放大，非缺陷。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import feature_cases  # noqa: E402

from miaosuan.core import features as F  # noqa: E402

pytestmark = pytest.mark.parity

_BASELINE = Path(__file__).resolve().parent.parent / "fixtures" / "features_baseline.npz"

#: helper_key -> 调用妙算 numpy helper 的函数
_HELPER_CALLS = {
    "ema_span12": lambda raw: F._ema_simple(raw["close"], 12),
    "ema_span15": lambda raw: F._ema_simple(raw["close"], 15),
    "ema_span20": lambda raw: F._ema_simple(raw["close"], 20),
    "ema_span26": lambda raw: F._ema_simple(raw["close"], 26),
    "ema_span50": lambda raw: F._ema_simple(raw["close"], 50),
    "ma10": lambda raw: F._ma(raw["close"], 10),
    "rolling_std20": lambda raw: F._rolling_std(raw["close"], 20),
    "atr14": lambda raw: F._atr(raw["close"], raw["high"], raw["low"], 14),
    "rvol": lambda raw: F._rvol(raw["close"]),
    "ac1": lambda raw: F._ac1(raw["close"]),
    "linear_slope20": lambda raw: F._linear_slope(raw["close"], 20),
    "trend_strength50": lambda raw: F._trend_strength(raw["close"], 50),
    "ts_corr10": lambda raw: F._ts_corr(raw["close"], raw["close"], 10),
    "robust_norm_close": lambda raw: F._robust_norm(raw["close"], 200),
    "trix15": lambda raw: F._trix(raw["close"], 15),
}

_HELPER_KEYS = [k for k, _ in feature_cases.HELPER_CASES]
_ATOL = 1e-5
_RTOL = 1e-5


@pytest.fixture(scope="session")
def baseline() -> dict:
    if not _BASELINE.is_file():
        pytest.skip(f"缺少 torch 基准：{_BASELINE}")
    with np.load(_BASELINE) as data:
        return {k: data[k] for k in data.files}


@pytest.fixture(scope="session")
def cases() -> dict:
    return feature_cases.build_cases()


def test_helper_keys_cover_baseline(baseline: dict) -> None:
    for key in _HELPER_KEYS:
        assert key in _HELPER_CALLS, f"helper 规格 {key} 缺少妙算调用"
        for case_name in feature_cases.build_cases():
            assert f"helper_{key}_{case_name}" in baseline, f"基准缺少 helper_{key}_{case_name}"


@pytest.mark.parametrize("helper_key", _HELPER_KEYS)
def test_helper_matches_torch(helper_key: str, cases: dict, baseline: dict) -> None:
    call = _HELPER_CALLS[helper_key]
    worst = ("", -1.0)
    for case_name, payload in cases.items():
        raw = {f: payload[f].astype(np.float32, copy=False) for f in feature_cases.FIELDS}
        actual = np.asarray(call(raw)).astype(np.float32, copy=False)
        expected = baseline[f"helper_{helper_key}_{case_name}"]
        assert actual.shape == expected.shape, (
            f"{helper_key}/{case_name} 形状 {actual.shape} != {expected.shape}"
        )
        a = actual.astype(np.float64)
        e = expected.astype(np.float64)
        tol = _ATOL + _RTOL * np.abs(e)
        d = np.abs(a - e)
        ratio = float((d / tol).max())
        if ratio > worst[1]:
            worst = (case_name, ratio)
    assert worst[1] <= 1.0, f"{helper_key} 超差：用例 {worst[0]} ratio={worst[1]:.3e}"
    print(f"  [helper {helper_key}] 最坏 ratio={worst[1]:.3e}（用例 {worst[0]}）")
