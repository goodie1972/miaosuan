# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""多因子组合（combine）：T1 信号层 + T2 ComboPayload。

规格 ``docs/combo-impl-spec.md``：

* §1.4 组合管线顺序与「禁仓位层平均」——数值断言盯守；
* §1.2 权重确定性（等权/IC 加权，禁 GA 搜权重）——本层只接收权重；
* §1.6 ComboPayload 在 spec 1.x 内以新 kind 兼容新增（不 bump 2.0）。
"""

from __future__ import annotations

import numpy as np
import pytest

from miaosuan.core.combo import combine_factors, combo_to_position, sign_align
from miaosuan.core.signal import compute_target_positions_stateless
from miaosuan.core.vm import StackVM
from miaosuan.ir.codec import decode_spec, encode_spec
from miaosuan.ir.schema import SPEC_VERSION, ComboPayload, StrategySpec


def _factor(seed: int, n: int = 600) -> np.ndarray:
    """造一条 [1, n] 的「已 z-score」因子（模拟 StackVM 输出尺度）。"""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, (1, n))
    return ((x - x.mean()) / (x.std() + 1e-8)).astype(np.float32)


# ── T1：符号对齐 ────────────────────────────────────────────────────────────


def test_sign_align_flips_negative_ic_member() -> None:
    """IC 为负的成员必须翻转，翻转后与正 IC 成员同向（相关变正）。"""
    a = _factor(1)
    b = (-a + 0.05 * _factor(2)).astype(np.float32)  # 与 a 反向
    assert np.corrcoef(a[0], b[0])[0, 1] < -0.9  # 前提：确实反向

    aligned = sign_align([a, b], [0.4, -0.35])
    assert aligned.shape == (2, 1, a.shape[1])
    assert np.corrcoef(aligned[0, 0], aligned[1, 0])[0, 1] > 0.9


def test_sign_align_keeps_direction_when_ic_is_zero_or_nan() -> None:
    """ic=0 / NaN 没有方向证据 → **保持原方向**，绝不凭空翻转。"""
    a = _factor(3)
    aligned = sign_align([a, a], [0.0, np.nan])
    np.testing.assert_array_equal(aligned[0], a)
    np.testing.assert_array_equal(aligned[1], a)


def test_sign_align_returns_float32_and_does_not_mutate_input() -> None:
    a = _factor(4)
    before = a.copy()
    aligned = sign_align([a], [-0.2])
    assert aligned.dtype == np.float32  # 下游按 float32 走，不许被 upcast
    np.testing.assert_array_equal(a, before)  # 纯函数


def test_sign_align_rejects_count_mismatch() -> None:
    with pytest.raises(ValueError, match="ic 数量与因子数量不一致"):
        sign_align([_factor(5)], [0.1, 0.2])
    with pytest.raises(ValueError, match="factors 不能为空"):
        sign_align([], [])


# ── T1：加权和 + 因果重标定 ─────────────────────────────────────────────────


def test_combine_factors_is_weighted_sum_then_restandardized() -> None:
    """先加权和、再因果重标定：输出应是 z-score 尺度（std ≈ 1）。"""
    a, b = _factor(6), _factor(7)
    agg = combine_factors([a, b], [1.0, 1.0], roll_window=50)
    assert agg.shape == a.shape
    assert agg.dtype == np.float32
    # 重标定后（去掉 warm-up）应为单位尺度
    assert 0.8 < float(agg[:, 50:].std()) < 1.2


def test_combine_factors_is_causal_no_lookahead() -> None:
    """重标定必须因果：改未来半段，过去半段输出逐点不变（无前视）。"""
    a, b = _factor(8), _factor(9)
    base = combine_factors([a, b], [1.0, 1.0], roll_window=50)

    b_perturbed = b.copy()
    b_perturbed[:, 400:] *= 3.0  # 只动 t>=400
    perturbed = combine_factors([a, b_perturbed], [1.0, 1.0], roll_window=50)

    np.testing.assert_array_equal(perturbed[:, :400], base[:, :400])


def test_combine_factors_weight_scale_is_irrelevant_after_restandardize() -> None:
    """整体缩放权重不影响结果（重标定对尺度不敏感）——等权语义稳健。"""
    a, b = _factor(10), _factor(11)
    x = combine_factors([a, b], [1.0, 1.0], roll_window=50)
    y = combine_factors([a, b], [7.0, 7.0], roll_window=50)
    np.testing.assert_allclose(y, x, rtol=1e-4, atol=1e-5)


def test_combine_factors_reuses_stackvm_normalization() -> None:
    """重标定必须与单因子栈**同口径**（复用 StackVM._normalize_output）。"""
    a, b = _factor(12), _factor(13)
    w = np.array([1.0, 1.0], dtype=np.float32)
    raw = np.tensordot(w, np.stack([a, b], axis=0), axes=(0, 0))
    expected = StackVM._normalize_output(np.asarray(raw, dtype=np.float32), 50)
    got = combine_factors([a, b], [1.0, 1.0], roll_window=50)
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-7)


def test_combine_factors_warmup_is_zero() -> None:
    """warm-up 期（前 roll_window-1 根）输出 0 —— 与单因子语义一致。"""
    agg = combine_factors([_factor(14)], [1.0], roll_window=100)
    np.testing.assert_array_equal(agg[:, :99], np.zeros((1, 99), dtype=np.float32))


def test_combine_factors_rejects_bad_inputs() -> None:
    a = _factor(15)
    with pytest.raises(ValueError, match="factors 不能为空"):
        combine_factors([], [1.0])
    with pytest.raises(ValueError, match="weights 数量与因子数量不一致"):
        combine_factors([a], [1.0, 2.0])
    with pytest.raises(ValueError, match="含非有限值"):
        combine_factors([a, np.full_like(a, np.nan)], [1.0, 1.0])
    with pytest.raises(ValueError, match="全为 0"):
        combine_factors([a, a], [0.0, 0.0])
    with pytest.raises(ValueError, match="形状"):
        combine_factors([a, _factor(16, n=599)], [1.0, 1.0])


# ── T1：仓位映射 ────────────────────────────────────────────────────────────


def test_combo_to_position_is_single_tanh_and_single_band() -> None:
    """单次 tanh + 单次中性带：与单因子信号映射**逐点一致**（禁仓位层平均）。"""
    agg = combine_factors([_factor(17), _factor(18)], [1.0, 1.0], roll_window=50)
    got = combo_to_position(agg, neutral_band=0.05, long_only=False)
    expected = compute_target_positions_stateless(agg, min_trade_exposure=0.05)
    np.testing.assert_array_equal(got, expected)
    # 中性带生效：|pos| < 0.05 全部归零
    raw_tanh = np.tanh(agg)
    assert np.all(np.abs(got[np.abs(raw_tanh) < 0.05]) == 0.0)


def test_combo_to_position_long_only_kills_negatives() -> None:
    agg = combine_factors([_factor(19)], [1.0], roll_window=50)
    pos = combo_to_position(agg, neutral_band=0.05, long_only=True)
    assert float(pos.min()) >= 0.0


def test_combo_to_position_rejects_negative_band() -> None:
    with pytest.raises(ValueError, match="neutral_band 非法"):
        combo_to_position(_factor(20), neutral_band=-0.01, long_only=False)


def test_position_layer_average_would_zero_members() -> None:
    """守护 §1.4：弱因子在**自己**的中性带里被清零 → 仓位层平均只剩 0。

    这正是「禁仓位层平均」的动机；若有人日后把实现改成先各自映射再平均，
    下面的断言会失败。
    """
    weak = (0.01 * _factor(21)).astype(np.float32)  # 幅度 < 中性带
    # 因子层：先合并再映射 → 有机会穿过中性带
    agg = combine_factors([weak, weak], [1.0, 1.0], roll_window=50)
    factor_level = combo_to_position(agg, neutral_band=0.05, long_only=False)
    # 仓位层：各自映射（全被自己的中性带清零）再平均 → 恒为 0
    p1 = combo_to_position(weak, neutral_band=0.05, long_only=False)
    p2 = combo_to_position((-weak), neutral_band=0.05, long_only=False)
    position_level = (p1 + p2) / 2.0

    assert float(np.abs(factor_level).max()) > 0.0, "因子层组合应产生非零仓位"
    np.testing.assert_array_equal(position_level, np.zeros_like(position_level))


# ── T2：ComboPayload ────────────────────────────────────────────────────────


def _combo_payload() -> ComboPayload:
    return ComboPayload(
        members=((33, 62, 3, 87), (72, 119, 73, 103)),
        weights=(0.5, 0.5),
        vocab_version="v9217a2c0d91a",
    )


def test_combo_payload_roundtrips_through_dict() -> None:
    p = _combo_payload()
    assert p.kind == "combo"
    assert p.n_members == 2
    assert ComboPayload.from_dict(p.to_dict()) == p
    assert p.to_dict() == {
        "kind": "combo",
        "members": [[33, 62, 3, 87], [72, 119, 73, 103]],
        "weights": [0.5, 0.5],
        "vocab_version": "v9217a2c0d91a",
    }


def test_combo_payload_spec_roundtrip_keeps_spec_id_and_version() -> None:
    """§1.6：spec 1.x 内以新 kind 兼容新增 —— SPEC_VERSION 不 bump。"""
    spec = StrategySpec(name="combo_h1", payload=_combo_payload())
    assert spec.spec_version == SPEC_VERSION == "1.0"

    restored = decode_spec(encode_spec(spec))
    assert isinstance(restored.payload, ComboPayload)
    assert restored.payload.members == spec.payload.members
    assert restored.payload.weights == spec.payload.weights
    # spec_id 内容寻址对新 kind 同样成立
    assert restored.spec_id == spec.spec_id


def test_combo_payload_tokens_property_is_empty() -> None:
    """``StrategySpec.tokens`` 只对因子体返回 token；组合体返回空元组（不误报）。"""
    spec = StrategySpec(name="combo_h1", payload=_combo_payload())
    assert spec.tokens == ()


def test_combo_payload_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError, match="members 不能为空"):
        ComboPayload(members=(), weights=(), vocab_version="v")
    with pytest.raises(ValueError, match="weights 数量与 members 不一致"):
        ComboPayload(members=((1, 2),), weights=(0.5, 0.5), vocab_version="v")
    with pytest.raises(ValueError, match="空 token 序列"):
        ComboPayload(members=((),), weights=(1.0,), vocab_version="v")
    with pytest.raises(ValueError, match="非有限"):
        ComboPayload(members=((1, 2),), weights=(float("nan"),), vocab_version="v")
    with pytest.raises(ValueError, match="全为 0"):
        ComboPayload(members=((1, 2),), weights=(0.0,), vocab_version="v")


def test_unknown_payload_kind_still_rejected() -> None:
    with pytest.raises(ValueError, match="未知 payload.kind"):
        StrategySpec.from_dict({"payload": {"kind": "bogus"}})
