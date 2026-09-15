# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""M4 特征数值对拍：妙算 numpy 实现 vs 冻结 AM 的 torch 基准。

基准文件由 ``scripts/gen_feature_baseline.py`` 在**真实 torch 环境**中用 AM 原始实现生成
（torch 2.14.0+cpu / numpy 2.5.3 / float32）：

* ``tests/fixtures/frozen_feature_inputs.npz`` —— 冻结输入面板；
* ``tests/fixtures/features_baseline.npz`` —— AM ``compute_features`` 输出 ``out_{case}``
  以及共享底层 helper 输出 ``helper_{key}_{case}``。

本测试在**无 torch** 的妙算 venv 中读取基准，用 numpy 实现复算，**逐特征**比对
``max|Δ|``。判定：``|Δ| <= atol + rtol*|expected|``（容差见 ``feature_cases``，按
「归一化放大」设定并附说明）。

注意：底层 helper 的**紧容差**对拍在 ``test_feature_helpers.py``；本文件聚焦 65 特征的
输出层对拍与维序/形状契约。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# 让 ``import feature_cases`` 可用（tests/parity 未做成包）
sys.path.insert(0, str(Path(__file__).resolve().parent))

import feature_cases  # noqa: E402

from miaosuan.core.features import (  # noqa: E402
    FEATURE_COUNT,
    FEATURE_NAMES,
    FEATURE_REGISTRY,
    compute_features,
)

pytestmark = pytest.mark.parity

_FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
_INPUTS = _FIXTURES / "frozen_feature_inputs.npz"
_BASELINE = _FIXTURES / "features_baseline.npz"
_INPUTS_META = _FIXTURES / "feature_inputs_meta.json"
_BASELINE_META = _FIXTURES / "features_baseline_meta.json"

_FIELDS = feature_cases.FIELDS


# ── 装置 ───────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def inputs() -> dict:
    if not _INPUTS.is_file():
        pytest.skip(f"缺少冻结输入：{_INPUTS}（先跑 scripts/gen_feature_baseline.py）")
    with np.load(_INPUTS) as data:
        return {k: data[k] for k in data.files}


@pytest.fixture(scope="session")
def baseline() -> dict:
    if not _BASELINE.is_file():
        pytest.skip(f"缺少 torch 基准：{_BASELINE}（先跑 scripts/gen_feature_baseline.py）")
    with np.load(_BASELINE) as data:
        return {k: data[k] for k in data.files}


@pytest.fixture(scope="session")
def inputs_meta() -> dict:
    if not _INPUTS_META.is_file():
        pytest.skip(f"缺少输入元信息：{_INPUTS_META}")
    return json.loads(_INPUTS_META.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def baseline_meta() -> dict:
    if not _BASELINE_META.is_file():
        pytest.skip(f"缺少基准元信息：{_BASELINE_META}")
    return json.loads(_BASELINE_META.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def cases() -> dict:
    return feature_cases.build_cases()


@pytest.fixture(scope="session")
def computed(cases: dict) -> dict:
    """各 case 的妙算输出 ``[N, 65, T]``（每 case 只算一次，供逐特征对拍复用）。"""
    return {
        case_name: compute_features({f: payload[f] for f in _FIELDS})
        for case_name, payload in cases.items()
    }


# ── 基准完整性 / 契约 ───────────────────────────────────────────────────────


def test_registry_default_is_65() -> None:
    assert FEATURE_COUNT == 65
    assert len(FEATURE_REGISTRY.feature_names) == 65


def test_baseline_covers_all_features(baseline_meta: dict) -> None:
    assert baseline_meta["feature_count"] == 65
    # 输出特征维顺序必须与妙算注册顺序逐项一致（axis order 契约）
    assert baseline_meta["feature_names"] == list(FEATURE_NAMES)


def test_inputs_meta_feature_order_matches(inputs_meta: dict) -> None:
    assert inputs_meta["feature_count"] == 65
    assert inputs_meta["feature_names"] == list(FEATURE_NAMES)


def test_baseline_inputs_match_cases(cases: dict, inputs: dict) -> None:
    """基准内保存的输入必须与本地用例逐位一致（防 RNG / numpy 版本漂移）。"""
    for case_name, payload in cases.items():
        for field in _FIELDS:
            saved = inputs[f"in_{case_name}_{field}"]
            local = payload[field]
            assert saved.shape == local.shape, f"{case_name}/{field} 形状不一致"
            assert np.array_equal(saved, local, equal_nan=True), (
                f"{case_name}/{field} 输入不一致（疑似 RNG 漂移）"
            )


def test_output_shape_and_axis_order(cases: dict, computed: dict) -> None:
    """输出形状 ``[N, F, T]``，``F == 65``，且 F 维顺序等于注册顺序。"""
    for case_name, payload in cases.items():
        actual = computed[case_name]
        n = payload["close"].shape[0]
        t = payload["close"].shape[1]
        assert actual.shape == (n, 65, t), f"{case_name} 形状 {actual.shape} != {(n, 65, t)}"


def test_output_has_no_nan_inf(computed: dict) -> None:
    """出口统一 nan_to_num，输出必须无 NaN/Inf。"""
    for case_name, actual in computed.items():
        assert np.isfinite(actual).all(), f"{case_name} 输出含 NaN/Inf"


# ── 逐特征对拍 ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("feat_idx", range(65), ids=list(FEATURE_NAMES))
def test_feature_matches_torch_baseline(
    feat_idx: int, cases: dict, computed: dict, baseline: dict
) -> None:
    name = FEATURE_NAMES[feat_idx]
    worst_case = ""
    worst_abs = -1.0

    for case_name, payload in cases.items():
        actual = computed[case_name][:, feat_idx, :].astype(np.float32, copy=False)
        expected = baseline[f"out_{case_name}"][:, feat_idx, :]

        assert actual.shape == expected.shape, (
            f"{name}/{case_name} 形状不一致：{actual.shape} vs {expected.shape}"
        )

        ok_mask, abs_diff = _compare(actual, expected, payload["atol"], payload["rtol"])
        n_bad = int((~ok_mask).sum())
        if n_bad:
            idx = np.argwhere(~ok_mask)[0]
            raise AssertionError(
                f"[{name}/{case_name}] {n_bad} 个元素超差；首个位置 {tuple(idx)}: "
                f"actual={actual[tuple(idx)]!r} expected={expected[tuple(idx)]!r} "
                f"|Δ|={abs_diff[tuple(idx)]!r} "
                f"(atol={payload['atol']}, rtol={payload['rtol']})"
            )

        finite = np.isfinite(actual) & np.isfinite(expected)
        if finite.any():
            m = float(abs_diff[finite].max())
            if m > worst_abs:
                worst_abs = m
                worst_case = case_name

    assert worst_abs >= 0.0
    print(f"  [{name}] 最大 |Δ|={worst_abs:.3e}（用例 {worst_case}）")


def test_feature_parity_global_worst(cases: dict, computed: dict, baseline: dict) -> None:
    """全局汇总：65 特征 × 6 用例，断言每点 ``|Δ| <= atol + rtol*|expected|``。

    以「误差/容差」比值作为最坏指标并打印最坏特征名（供验收报告）。
    """
    worst = ("", "", -1.0, -1.0)  # feature, case, max_abs, max_ratio
    total = 0
    for case_name, payload in cases.items():
        actual_all = computed[case_name]
        expected_all = baseline[f"out_{case_name}"]
        for fi, name in enumerate(FEATURE_NAMES):
            actual = actual_all[:, fi, :].astype(np.float32, copy=False)
            expected = expected_all[:, fi, :]
            ok_mask, abs_diff = _compare(actual, expected, payload["atol"], payload["rtol"])
            assert ok_mask.all(), f"[{name}/{case_name}] 超差"

            tol = payload["atol"] + payload["rtol"] * np.abs(expected.astype(np.float64))
            finite = np.isfinite(actual) & np.isfinite(expected)
            total += 1
            if finite.any():
                ratio = float((abs_diff[finite] / tol[finite]).max())
                m_abs = float(abs_diff[finite].max())
                if ratio > worst[3]:
                    worst = (name, case_name, m_abs, ratio)

    assert worst[3] <= 1.0, f"全局最大误差/容差比超限：{worst}"
    print(
        f"\n全局最坏：ratio={worst[3]:.3e} |Δ|={worst[2]:.3e} "
        f"（特征 {worst[0]} / 用例 {worst[1]}），比较组数 {total}"
    )


# ── 参数注入（active_features 白名单，取代 AM 的 import 期 IO）────────────


def test_active_features_param_injection(cases: dict) -> None:
    payload = cases["base"]
    raw = {f: payload[f] for f in _FIELDS}
    full = compute_features(raw)

    subset = ["RET", "VOL_Z", "CS_RANK_RET5"]
    sub = compute_features(raw, active_features=subset)
    idx = [FEATURE_NAMES.index(n) for n in subset]
    assert sub.shape == (full.shape[0], len(subset), full.shape[2])
    for j, i in enumerate(idx):
        assert np.array_equal(sub[:, j, :], full[:, i, :]), f"{subset[j]} 列不一致"


def test_active_features_order_follows_defs(cases: dict) -> None:
    """白名单按 _FEATURE_DEFS 原始顺序过滤，维序稳定（与传入顺序无关）。"""
    payload = cases["base"]
    raw = {f: payload[f] for f in _FIELDS}
    a = compute_features(raw, active_features=["VOL_Z", "RET"])
    b = compute_features(raw, active_features=["RET", "VOL_Z"])
    assert np.array_equal(a, b)
    # 注册顺序 RET(0) 在前，VOL_Z(15) 在后
    assert np.array_equal(a[:, 0, :], compute_features(raw)[:, FEATURE_NAMES.index("RET"), :])


# ── 比较工具（与 ops 对拍一致的语义）──────────────────────────────────────


def _compare(
    actual: np.ndarray, expected: np.ndarray, atol: float, rtol: float
) -> tuple[np.ndarray, np.ndarray]:
    """逐元素比较，返回 (ok_mask, abs_diff)。

    处理规则：两侧同为 NaN -> 相等；两侧同为同号 Inf -> 相等；两侧均为有限值 ->
    ``|Δ| <= atol + rtol*|expected|``；其余 -> 不相等。
    """
    a = actual.astype(np.float64)
    e = expected.astype(np.float64)

    both_nan = np.isnan(a) & np.isnan(e)
    both_inf = np.isinf(a) & np.isinf(e) & (np.sign(a) == np.sign(e))
    finite = np.isfinite(a) & np.isfinite(e)

    abs_diff = np.full(e.shape, np.inf, dtype=np.float64)
    abs_diff[finite] = np.abs(a[finite] - e[finite])
    tol = atol + rtol * np.abs(e)
    ok = both_nan | both_inf | (finite & (abs_diff <= tol))
    return ok, abs_diff
