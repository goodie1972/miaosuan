# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""M5 StackVM 数值对拍：妙算 numpy 实现 vs 冻结 AM 的 torch 基准。

覆盖（team-lead M5 难点 1）：

* ``_normalize_output`` —— 滚动 500（warm-up 499 → 0）/ expanding / 截面 / 常数短路；
* ``StackVM.execute`` —— 合法公式 / 参数不足 / 栈深非法（残留≠1 / 空）/ 特征越界 / 未知算子
  / 截断特征张量 / NaN-Inf（``nan_to_num``）。

基准由 ``scripts/gen_m5_baseline.py`` 在真实 torch 环境用 AM 原始实现生成。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import m5_cases  # noqa: E402

from miaosuan.core.vm import StackVM  # noqa: E402
from miaosuan.core.vocab import FORMULA_VOCAB, VOCAB_VERSION  # noqa: E402

pytestmark = pytest.mark.parity

# 容差：特征/算子层 float32 归约顺序噪声经 JUMP(expanding zscore)+归一化放大
_VM_ATOL = 5e-3
_VM_RTOL = 1e-2
_NORM_ATOL = 1e-5
_NORM_RTOL = 1e-5


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if a.size else 0.0


# ── 契约 ───────────────────────────────────────────────────────────────────


def test_vocab_version_frozen() -> None:
    assert VOCAB_VERSION == "v9217a2c0d91a"
    assert FORMULA_VOCAB.size == 127
    assert FORMULA_VOCAB.operator_offset == 65


def test_meta_covers_all_vm_keys(m5_meta: dict) -> None:
    assert len(m5_meta["vm_keys"]) == len(m5_cases.VM_FORMULAS) * len(
        m5_cases.build_feat_panels()
    )


# ── _normalize_output 对拍 ─────────────────────────────────────────────────


@pytest.mark.parametrize("name", m5_cases.build_normalize_arrays().keys())
def test_normalize_output_parity(name: str, m5_npz: dict) -> None:
    x = m5_npz[f"norm_in__{name}"]
    got = StackVM._normalize_output(x, 500)
    exp = m5_npz[f"norm_out__{name}"]
    assert got.shape == exp.shape
    d = _max_abs(got, exp)
    assert d <= _NORM_ATOL + _NORM_RTOL * max(1.0, float(np.max(np.abs(exp)))), (
        f"normalize/{name} max|Δ|={d:.3e} 超限"
    )


def test_normalize_rolling_warmup_zero(m5_npz: dict) -> None:
    """滚动路径：前 499 根必须为 0（warm-up），第 500 根起为真实 z-score。"""
    for name in ("n1_t500", "n1_t800"):
        x = m5_npz[f"norm_in__{name}"]
        got = StackVM._normalize_output(x, 500)
        assert np.all(got[:, :499] == 0.0), f"{name}: warm-up 段非 0"
        assert np.any(got[:, 499:] != 0.0), f"{name}: warm-up 后全 0（异常）"


def test_normalize_global_constant_short_circuit(m5_npz: dict) -> None:
    """全局常数（std < 1e-6）原样返回。"""
    for name in ("const_n1", "const_n2"):
        x = m5_npz[f"norm_in__{name}"]
        got = StackVM._normalize_output(x, 500)
        assert np.array_equal(got, x), f"{name}: 常数短路失效"


def test_normalize_clip_bound() -> None:
    """归一化输出必须落在 [-3, 3]。"""
    rng = np.random.default_rng(123)
    for shape in ((1, 800), (5, 300)):
        x = rng.normal(0, 1, shape).astype(np.float32)
        got = StackVM._normalize_output(x, 500)
        assert float(got.min()) >= -3.0 - 1e-6
        assert float(got.max()) <= 3.0 + 1e-6


# ── StackVM.execute 对拍 ───────────────────────────────────────────────────


def _execute(panel: np.ndarray, formula_name: str) -> np.ndarray | None:
    vm = StackVM()
    return vm.execute(list(m5_cases.VM_FORMULAS[formula_name]), panel)


def test_vm_none_cases(m5_npz: dict, m5_meta: dict) -> None:
    """基准中为 None 的组合，妙算必须同样返回 None（不可求值判定不变）。"""
    none_keys = set(m5_meta["vm_none_keys"])
    assert none_keys, "基准未记录任何 None 组合（用例失效？）"
    for key in sorted(none_keys):
        panel_name, formula_name = key.split("__")
        got = _execute(m5_npz[f"vmfeat__{panel_name}"], formula_name)
        assert got is None, f"{key} 期望 None，实际得到 {None if got is None else got.shape}"


def test_vm_array_cases(m5_npz: dict, m5_meta: dict) -> None:
    """非 None 组合逐点对拍 max|Δ|。"""
    none_keys = set(m5_meta["vm_none_keys"])
    worst = 0.0
    worst_key = ""
    checked = 0
    for key in m5_meta["vm_keys"]:
        if key in none_keys:
            continue
        panel_name, formula_name = key.split("__")
        got = _execute(m5_npz[f"vmfeat__{panel_name}"], formula_name)
        assert got is not None, f"{key} 期望数组，实际为 None"
        exp = m5_npz[f"vmout__{key}"]
        assert got.shape == exp.shape
        d = _max_abs(got, exp)
        if d > worst:
            worst = d
            worst_key = key
        checked += 1
    assert checked > 0
    assert worst <= _VM_ATOL + _VM_RTOL, f"VM 对拍最坏 {worst:.3e} @ {worst_key}"


# ── 四类判定显式校验（语义 sentinel）────────────────────────────────────────


def test_execute_semantics_explicit() -> None:
    feat = m5_cases.build_feat_panels()["cs"]
    vm = StackVM()

    # ① 合法公式 → [N, T]
    ok = vm.execute([0, 1, 65], feat)
    assert ok is not None and ok.shape == (feat.shape[0], feat.shape[2])

    # ② 参数不足（ADD 需 2，操作数仅 1）→ None
    assert vm.execute([65], feat) is None

    # ③ 栈深非法：残留 2 / 空公式 → None
    assert vm.execute([0, 1], feat) is None
    assert vm.execute([], feat) is None

    # ④ 未知算子 token → None
    assert vm.execute([200], feat) is None
    # 特征 token 越界（截断到 10 通道后引用 f33）→ None
    assert vm.execute([33], feat[:, :10, :]) is None


def test_execute_nan_inf_nan_to_num() -> None:
    """NaN/Inf 经算子后按 AM 语义替换：nan→0, +inf→1, -inf→-1。"""
    feat = np.zeros((1, 65, 300), dtype=np.float32)
    feat[:, 0, :] = np.inf
    feat[:, 1, :] = -np.inf
    feat[:, 2, :] = np.nan
    vm = StackVM()
    # DECAY(inf) = inf → nan_to_num → 1.0（常数输出，短路返回）
    r_inf = vm.execute([0, 74], feat)
    assert r_inf is not None and np.allclose(r_inf, 1.0)
    # inf / (-inf + eps) = nan → nan_to_num(nan=0) → 0.0
    r_nan = vm.execute([0, 1, 68], feat)
    assert r_nan is not None and np.allclose(r_nan, 0.0)
