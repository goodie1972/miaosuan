# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""M3 算子数值对拍：妙算 numpy 实现 vs 冻结 AM 的 torch 基准。

基准文件 ``tests/fixtures/ops_baseline.npz`` 由 ``scripts/gen_ops_baseline.py`` 在
**真实 torch 环境**中用 AM 原始实现生成（torch 2.14.0+cpu / numpy 2.5.3 / float32）。
本测试在**无 torch** 的妙算 venv 中读取基准，用妙算 numpy 实现复算，逐算子逐用例比对。

判定：``|Δ| <= atol + rtol * |expected|``（float32 输入，atol/rtol 见 ``ops_cases``）。
NaN 用例中，依赖排序（argmax/argsort）的算子因两者 NaN 排位规则未承诺一致而不参与比对
（``ops_cases.NAN_CASE_EXEMPT_OPS``），其余算子要求 NaN 位置也一致。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# 让 ``import ops_cases`` 可用（tests/parity 未做成包）
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ops_cases  # noqa: E402

from miaosuan.core.ops import OPS_CONFIG  # noqa: E402

pytestmark = pytest.mark.parity

_FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
_BASELINE = _FIXTURES / "ops_baseline.npz"
_META = _FIXTURES / "ops_baseline_meta.json"

#: op 名 -> (transform, arity)
_OPS: dict[str, tuple] = {name: (fn, arity) for name, fn, arity in OPS_CONFIG}


# ── 装置 ───────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def baseline() -> dict:
    if not _BASELINE.is_file():
        pytest.skip(f"缺少 torch 基准：{_BASELINE}（先跑 scripts/gen_ops_baseline.py）")
    with np.load(_BASELINE) as data:
        return {k: data[k] for k in data.files}


@pytest.fixture(scope="session")
def meta() -> dict:
    if not _META.is_file():
        pytest.skip(f"缺少基准元信息：{_META}")
    return json.loads(_META.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def cases() -> dict:
    return ops_cases.build_cases()


# ── 基准完整性 ─────────────────────────────────────────────────────────────


def test_baseline_covers_all_operators(meta: dict) -> None:
    assert meta["operator_count"] == 62
    assert meta["operator_names"] == [name for name, _f, _a in OPS_CONFIG]


def test_baseline_inputs_match_cases(cases: dict, baseline: dict) -> None:
    """基准内保存的输入必须与本地用例逐位一致（防 RNG / numpy 版本漂移）。"""
    for case_name, payload in cases.items():
        for key in ("x", "y", "z"):
            saved = baseline[f"in_{case_name}_{key}"]
            local = payload[key]
            assert saved.shape == local.shape, f"{case_name}/{key} 形状不一致"
            same = np.array_equal(saved, local, equal_nan=True)
            assert same, f"{case_name}/{key} 输入不一致（疑似 RNG 漂移）"


# ── 逐算子对拍 ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("op_name", sorted(_OPS))
def test_operator_matches_torch_baseline(op_name: str, cases: dict, baseline: dict) -> None:
    fn, arity = _OPS[op_name]
    worst_case = ""
    worst_abs = -1.0

    for case_name, payload in cases.items():
        if case_name == "nan" and op_name in ops_cases.NAN_CASE_EXEMPT_OPS:
            continue

        operands = [payload[k] for k in ("x", "y", "z")[:arity]]
        actual = np.asarray(fn(*operands)).astype(np.float32, copy=False)
        expected = baseline[f"out_{op_name}_{case_name}"]

        assert actual.shape == expected.shape, (
            f"{op_name}/{case_name} 形状不一致：{actual.shape} vs {expected.shape}"
        )

        ok_mask, abs_diff = _compare(actual, expected, payload["atol"], payload["rtol"])
        n_bad = int((~ok_mask).sum())
        if n_bad:
            idx = np.argwhere(~ok_mask)[0]
            raise AssertionError(
                f"[{op_name}/{case_name}] {n_bad} 个元素超差；"
                f"首个位置 {tuple(idx)}: actual={actual[tuple(idx)]!r} "
                f"expected={expected[tuple(idx)]!r} |Δ|={abs_diff[tuple(idx)]!r} "
                f"(atol={payload['atol']}, rtol={payload['rtol']})"
            )

        finite = np.isfinite(actual) & np.isfinite(expected)
        if finite.any():
            m = float(abs_diff[finite].max())
            if m > worst_abs:
                worst_abs = m
                worst_case = case_name

    # 记录每算子最坏误差（-v 时可见）
    assert worst_abs >= 0.0
    print(f"  [{op_name}] 最大 |Δ|={worst_abs:.3e}（用例 {worst_case}）")


def test_ops_parity_global_worst(cases: dict, baseline: dict) -> None:
    """全局汇总：所有算子 × 所有用例，断言每点 ``|Δ| <= atol + rtol*|expected|``。

    以「误差/容差」比值作为最坏指标（float32 在 ``extreme`` 用例上绝对误差可达
    ~1e2 量级，但相对量级仍在容差内，属正常浮点精度，非移植缺陷）。
    """
    worst = ("", "", -1.0, -1.0)  # op, case, max_abs, max_ratio
    total = 0
    for op_name, (fn, arity) in _OPS.items():
        for case_name, payload in cases.items():
            if case_name == "nan" and op_name in ops_cases.NAN_CASE_EXEMPT_OPS:
                continue
            operands = [payload[k] for k in ("x", "y", "z")[:arity]]
            actual = np.asarray(fn(*operands)).astype(np.float32, copy=False)
            expected = baseline[f"out_{op_name}_{case_name}"]
            ok_mask, abs_diff = _compare(actual, expected, payload["atol"], payload["rtol"])
            assert ok_mask.all(), f"[{op_name}/{case_name}] 超差"

            tol = payload["atol"] + payload["rtol"] * np.abs(expected.astype(np.float64))
            finite = np.isfinite(actual) & np.isfinite(expected)
            total += 1
            if finite.any():
                ratio = float((abs_diff[finite] / tol[finite]).max())
                m_abs = float(abs_diff[finite].max())
                if ratio > worst[3]:
                    worst = (op_name, case_name, m_abs, ratio)

    assert worst[3] <= 1.0, f"全局最大误差/容差比超限：{worst}"
    print(
        f"\n全局最坏：ratio={worst[3]:.3e} |Δ|={worst[2]:.3e} "
        f"（算子 {worst[0]} / 用例 {worst[1]}），比较组数 {total}"
    )


# ── 比较工具 ───────────────────────────────────────────────────────────────


def _compare(
    actual: np.ndarray, expected: np.ndarray, atol: float, rtol: float
) -> tuple[np.ndarray, np.ndarray]:
    """逐元素比较，返回 (ok_mask, abs_diff)。

    处理规则：
      * 两侧同为 NaN        -> 视为相等；
      * 两侧同为同号 Inf    -> 视为相等；
      * 两侧均为有限值      -> ``|Δ| <= atol + rtol*|expected|``；
      * 其余（一侧 NaN/Inf）-> 不相等。
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
