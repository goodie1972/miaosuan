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

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from typer.testing import CliRunner

from miaosuan.cli import app
from miaosuan.config import AppConfig, ConfigError
from miaosuan.core.combo import combine_factors, combo_to_position, sign_align
from miaosuan.core.signal import compute_target_positions_stateless
from miaosuan.core.vm import StackVM
from miaosuan.core.vocab import VOCAB_VERSION
from miaosuan.data.loader import load
from miaosuan.data.split import HoldoutSealRegistry
from miaosuan.ir.codec import decode_spec, encode_spec
from miaosuan.ir.schema import SPEC_VERSION, ComboPayload, StrategySpec
from miaosuan.pipeline import _select_members, _sum_run_n_trials, run_combine
from miaosuan.search.mine import _gate_candidate
from miaosuan.search.rpn import Individual, random_feasible

_runner = CliRunner()


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


# ══════════════════════════════════════════════════════════════════════════
# T3 pipeline.run_combine + T4 CLI combine（规格 §3 T3/T4、§5 测试清单）
# ══════════════════════════════════════════════════════════════════════════



def _feasible(count: int, seed: int = 0) -> list[list[int]]:
    """造 ``count`` 条**互不相同**的可行 8-token 公式。"""
    rng = np.random.default_rng(seed)
    out: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    while len(out) < count:
        toks = [int(t) for t in random_feasible(rng, 8)]
        key = tuple(toks)
        if key in seen:
            continue
        seen.add(key)
        out.append(toks)
    return out


def _write_candidates(
    path: Path,
    tokens_list: list[list[int]],
    *,
    timeframe: str = "H1",
    vocab_version: str = VOCAB_VERSION,
    data_fingerprint: str = "f" * 64,
    n_trials: int = 1000,
    spec_id: str = "deadbeef",
    with_top_level_trials: bool = True,
    with_fingerprint: bool = True,
) -> Path:
    """落一份**结构与 mine --dump-candidates 一致**的候选 JSON。"""
    entries = [
        {
            "rank": rank,
            "tokens": toks,
            "decoded": f"member_{rank}",
            "train_score": 1.0,
            "val_score": 1.0,
            "status": "ok",
            "cost_sensitivity": {"1": 1.0},
            "sharpe_2x": 1.0,
            "wf_win_rate": 1.0,
            "dsr": 0.1,
            "verdict": "BLOCKED",
            # 顶层缺 n_trials 时的兜底来源（规格 §1.3 指定的字段）
            "verdict_snapshot": json.dumps({"n_trials": n_trials, "status": "BLOCKED"}),
        }
        for rank, toks in enumerate(tokens_list)
    ]
    doc: dict[str, Any] = {
        "version": 1,
        "spec": "artifacts/spec.json",
        "spec_id": spec_id,
        "budget": "quick",
        "symbol": "XAUUSD",
        "timeframe": timeframe,
        "market": "FOREX_XAUUSD",
        "git_sha": "deadbeef",
        "vocab_version": vocab_version,
        "semantics": {
            "timeframe": timeframe,
            "position_fn": "tanh",
            "neutral_band": 0.05,
            "long_short": True,
        },
        "n_candidates": len(entries),
        "candidates": entries,
    }
    if with_top_level_trials:
        doc["n_trials"] = n_trials
    if with_fingerprint:
        doc["data_fingerprint"] = data_fingerprint
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _csv(tmp_path: Path, name: str = "XAUUSD_H1.csv", n: int = 1000) -> Path:
    """造一份可 ``load`` 的 CSV 行情（确定性随机游走）。"""
    import pandas as pd

    rng = np.random.default_rng(11)
    steps = rng.normal(0.0, 0.002, size=n)
    close = 2000.0 * np.exp(np.cumsum(steps))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    frame = pd.DataFrame(
        {
            "time": 1_700_000_000 + np.arange(n) * 3600,
            "open": open_,
            "high": np.maximum(open_, close) * 1.001,
            "low": np.minimum(open_, close) * 0.999,
            "close": close,
            "volume": np.full(n, 1000.0),
        }
    )
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return path


# ── T3：入参与校验拒绝 ───────────────────────────────────────────────────────


def test_run_combine_rejects_unknown_method(tmp_path: Path) -> None:
    toks = _feasible(3)
    a = _write_candidates(tmp_path / "a.json", toks)
    with pytest.raises(ConfigError, match="未知权重方法"):
        run_combine([a], panel=load(_csv(tmp_path)), config=AppConfig(), method="ga")


def test_run_combine_rejects_bad_top_per_file(tmp_path: Path) -> None:
    a = _write_candidates(tmp_path / "a.json", _feasible(3))
    with pytest.raises(ConfigError, match="top_per_file"):
        run_combine(
            [a], panel=load(_csv(tmp_path)), config=AppConfig(), top_per_file=0
        )


def test_run_combine_rejects_missing_candidates_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="候选文件不存在"):
        run_combine(
            [tmp_path / "nope.json"],
            panel=load(_csv(tmp_path)),
            config=AppConfig(),
        )


def test_run_combine_rejects_timeframe_mismatch(tmp_path: Path) -> None:
    """同周期校验：候选周期必须与 ``--data`` 文件名推断出的周期一致（§3）。"""
    toks = _feasible(3)
    a = _write_candidates(tmp_path / "a.json", toks, timeframe="H1")
    b = _write_candidates(tmp_path / "b.json", _feasible(3, seed=5), timeframe="M15")
    with pytest.raises(ConfigError, match="不一致"):
        run_combine([a, b], panel=load(_csv(tmp_path)), config=AppConfig())


def test_run_combine_rejects_untrusted_timeframe(tmp_path: Path) -> None:
    """文件名推不出周期（配置默认不可信）→ 拒绝，而不是退回 H1 猜一个（§3）。"""
    a = _write_candidates(tmp_path / "a.json", _feasible(3))
    with pytest.raises(ConfigError, match="推不出周期"):
        run_combine([a], panel=load(_csv(tmp_path, name="XAUUSD.csv")), config=AppConfig())


def test_run_combine_rejects_vocab_mismatch(tmp_path: Path) -> None:
    a = _write_candidates(tmp_path / "a.json", _feasible(3))
    b = _write_candidates(
        tmp_path / "b.json", _feasible(3, seed=5), vocab_version="v_OLD_OTHER"
    )
    with pytest.raises(ConfigError, match="词表版本不一致"):
        run_combine([a, b], panel=load(_csv(tmp_path)), config=AppConfig())


def test_run_combine_rejects_foreign_vocab(tmp_path: Path) -> None:
    """词表与当前版本不同 → token 无意义，直接拒绝（§3 同 vocab 延伸）。"""
    a = _write_candidates(
        tmp_path / "a.json", _feasible(3), vocab_version="v_NOT_CURRENT"
    )
    with pytest.raises(ConfigError, match="词表 .* 不一致"):
        run_combine([a], panel=load(_csv(tmp_path)), config=AppConfig())


def test_run_combine_rejects_fingerprint_mismatch(tmp_path: Path) -> None:
    a = _write_candidates(tmp_path / "a.json", _feasible(3), data_fingerprint="a" * 64)
    b = _write_candidates(
        tmp_path / "b.json", _feasible(3, seed=5), data_fingerprint="b" * 64
    )
    with pytest.raises(ConfigError, match="指纹不一致"):
        run_combine([a, b], panel=load(_csv(tmp_path)), config=AppConfig())


def test_run_combine_rejects_missing_fingerprint(tmp_path: Path) -> None:
    """``None == None`` 陷阱：两份候选都没指纹也必须拒绝，不能静默通过（§3）。"""
    a = _write_candidates(tmp_path / "a.json", _feasible(3), with_fingerprint=False)
    b = _write_candidates(
        tmp_path / "b.json", _feasible(3, seed=5), with_fingerprint=False
    )
    with pytest.raises(ConfigError, match="缺 data_fingerprint"):
        run_combine([a, b], panel=load(_csv(tmp_path)), config=AppConfig())


def test_run_combine_rejects_single_distinct_member(tmp_path: Path) -> None:
    """两份候选去重后只剩 1 条 → 不是「组合」，拒绝（§1 要多因子）。"""
    one = _feasible(1)[0]
    a = _write_candidates(tmp_path / "a.json", [one])
    b = _write_candidates(tmp_path / "b.json", [one], spec_id="other")
    with pytest.raises(ConfigError, match="至少需要 2"):
        run_combine([a, b], panel=load(_csv(tmp_path)), config=AppConfig())


# ── T3：seal=False 不触发封印（§1.5 / §5）─────────────────────────────────


def test_run_combine_never_seals_holdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``make_split(..., seal=False)``：组合**不得**写 hold-out 封印台账。"""

    def _boom(self: Any, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("combine 不应调用 registry.seal")

    monkeypatch.setattr(HoldoutSealRegistry, "seal", _boom)
    a = _write_candidates(tmp_path / "a.json", _feasible(3))
    b = _write_candidates(tmp_path / "b.json", _feasible(3, seed=5))
    outcome = run_combine([a, b], panel=load(_csv(tmp_path)), config=AppConfig())
    # 走到这里说明 seal 一次都没被调用；顺带断言 hold-out 未消费
    assert outcome.spec.evidence.holdout_sharpe is None


# ── T3：n_trials 按运行去重累加（§1.3 / §5）───────────────────────────────


def test_n_trials_counts_each_run_once(tmp_path: Path) -> None:
    """同一运行的两个候选只计一次该运行的 n_trials；不同运行各自累加。"""
    toks = _feasible(4)
    a = _write_candidates(tmp_path / "a.json", toks[:3], n_trials=1000)
    b = _write_candidates(tmp_path / "b.json", toks[1:] + [toks[3]], n_trials=500, spec_id="b")

    def _docs(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    docs = [(a, _docs(a)), (b, _docs(b))]
    members, runs = _select_members(docs, top_per_file=3)
    # a 贡献 3 条，b 的前两条与 a 重复被去重，只有第 3 条是新的
    assert len(members) == 4
    assert len(runs) == 2
    assert _sum_run_n_trials(docs, runs) == 1500


def test_n_trials_dedups_by_run_path(tmp_path: Path) -> None:
    """同一文件被传两次 → 只算一次，Σ 不虚高（否则 DSR 被过度压低）。"""
    toks = _feasible(3)
    a = _write_candidates(tmp_path / "a.json", toks, n_trials=1000)
    outcome = run_combine(
        [a, a],
        panel=load(_csv(tmp_path)),
        config=AppConfig(),
        created_at="2026-01-01T00:00:00Z",
    )
    assert outcome.spec.evidence.n_trials == 1000
    assert outcome.candidate.n_trials == 1000
    # sources 只应有 1 个（去重后）
    assert len(outcome.spec.provenance.config_snapshot["combo"]["sources"]) == 1


def test_n_trials_falls_back_to_verdict_snapshot(tmp_path: Path) -> None:
    """旧格式（顶层缺 n_trials）退回 ``verdict_snapshot.n_trials``，而不是报错/记 0。"""
    toks = _feasible(3)
    a = _write_candidates(tmp_path / "a.json", toks, with_top_level_trials=False)
    b = _write_candidates(
        tmp_path / "b.json", _feasible(3, seed=5), n_trials=700, spec_id="b"
    )
    outcome = run_combine(
        [a, b], panel=load(_csv(tmp_path)), config=AppConfig()
    )
    # a=1000（来自 verdict_snapshot）、b=700（顶层）
    assert outcome.spec.evidence.n_trials == 1700


def test_run_combine_dedups_tokens_but_counts_both_runs(tmp_path: Path) -> None:
    """**成员**去重不影响**运行**计数：同一条公式被两运行各自挖到，两次搜索都算。"""
    toks = _feasible(3)
    a = _write_candidates(tmp_path / "a.json", toks, n_trials=1000)
    b = _write_candidates(tmp_path / "b.json", toks, n_trials=500, spec_id="b")
    outcome = run_combine([a, b], panel=load(_csv(tmp_path)), config=AppConfig())
    # 成员去重后只剩 3 条，但两个运行的 n_trials 都要计
    assert outcome.candidate.n_members == 3
    assert outcome.spec.evidence.n_trials == 1500


# ── T3：门禁快照与单因子口径一致（§1.7 / §5）───────────────────────────────


def _single_factor_gate(tokens: list[int]) -> tuple[Any, dict[str, Any]]:
    """用单因子路径跑一次门禁，返回 ``(verdict, 快照 dict)``（口径对照基准）。"""
    from miaosuan.core.features import compute_features

    rng = np.random.default_rng(11)
    n = 1000
    steps = rng.normal(0.0, 0.002, size=n)
    close = 2000.0 * np.exp(np.cumsum(steps))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    raw = {
        "open": open_.reshape(1, -1).astype(np.float32),
        "high": (np.maximum(open_, close) * 1.001).reshape(1, -1).astype(np.float32),
        "low": (np.minimum(open_, close) * 0.999).reshape(1, -1).astype(np.float32),
        "close": close.reshape(1, -1).astype(np.float32),
        "volume": np.full((1, n), 1000.0, dtype=np.float32),
    }
    features = compute_features(raw)
    target = np.zeros((1, n), dtype=np.float32)
    op = raw["open"]
    target[:, :-2] = np.log(op[:, 2:] / op[:, 1:-1])
    from miaosuan.search.ga import AMFitnessEvaluator

    evaluator = AMFitnessEvaluator(
        features, target, cost_rate=0.0003, periods_per_year=6240, n_folds=5, gap=20
    )
    ind = Individual(tokens=np.asarray(tokens, dtype=np.int64))
    ind.train_score = 1.0
    ind.val_score = 1.0
    candidate = _gate_candidate(
        ind, evaluator, base_cost_rate=0.0003, periods_per_year=6240, n_trials=1000
    )
    return candidate, json.loads(candidate.verdict_snapshot)


def test_gate_snapshot_keys_match_single_factor() -> None:
    """组合门禁快照的**字段集与单因子逐键相同**（§1.7 同款 evaluate_verdict）。"""
    tokens = _feasible(3)
    single_candidate, single_snap = _single_factor_gate(tokens[0])
    combo_snap = json.loads(
        _combo_candidate(tokens[:3], method="equal").verdict_snapshot
    )
    assert set(combo_snap) == set(single_snap), (
        f"组合快照字段与单因子不一致：多 {set(combo_snap) - set(single_snap)}，"
        f"少 {set(single_snap) - set(combo_snap)}"
    )
    # 双方都是 BLOCKED（随机数据 DSR 必然过低），失败原因词表也必须一致
    assert single_candidate.verdict == "BLOCKED"
    assert combo_snap["status"] == "BLOCKED"
    assert single_snap["hard_failures"] and combo_snap["hard_failures"]
    assert set(combo_snap["hard_failures"]) <= {
        "val_score<=0",
        "sharpe_2x<=0",
        "dsr<0.5",
    }


def test_gate_dsr_uses_accumulated_n_trials(tmp_path: Path) -> None:
    """n_trials 越大 DSR 越小：证明 combine 把 Σ 值真的喂进了 DSR（§1.3）。"""
    tokens = _feasible(3)
    a = _write_candidates(tmp_path / "a.json", tokens, n_trials=100)
    b = _write_candidates(tmp_path / "b.json", _feasible(3, seed=5), n_trials=100, spec_id="b")
    small = run_combine([a], panel=load(_csv(tmp_path)), config=AppConfig())
    big = run_combine(
        [a, b],
        panel=load(_csv(tmp_path)),
        config=AppConfig(),
        top_per_file=1,
    )
    assert big.spec.evidence.n_trials > small.spec.evidence.n_trials
    assert big.candidate.dsr <= small.candidate.dsr


def _combo_candidate(tokens: list[list[int]], *, method: str = "equal") -> Any:
    """直接调 :func:`miaosuan.search.combo.combine`（不落盘，供门禁口径对照）。"""
    from miaosuan.core.features import compute_features
    from miaosuan.search.combo import combine
    from miaosuan.search.ga import AMFitnessEvaluator

    rng = np.random.default_rng(11)
    n = 1000
    steps = rng.normal(0.0, 0.002, size=n)
    close = 2000.0 * np.exp(np.cumsum(steps))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    raw = {
        "open": open_.reshape(1, -1).astype(np.float32),
        "high": (np.maximum(open_, close) * 1.001).reshape(1, -1).astype(np.float32),
        "low": (np.minimum(open_, close) * 0.999).reshape(1, -1).astype(np.float32),
        "close": close.reshape(1, -1).astype(np.float32),
        "volume": np.full((1, n), 1000.0, dtype=np.float32),
    }
    features = compute_features(raw)
    target = np.zeros((1, n), dtype=np.float32)
    op = raw["open"]
    target[:, :-2] = np.log(op[:, 2:] / op[:, 1:-1])
    evaluator = AMFitnessEvaluator(
        features, target, cost_rate=0.0003, periods_per_year=6240, n_folds=5, gap=20
    )
    return combine(tokens, evaluator=evaluator, n_trials=1000, method=method)


# ── T4：CLI combine 端到端（§5「CLI 集成」）───────────────────────────────────


def test_combine_cli_end_to_end(tmp_path: Path) -> None:
    """构造两份小 candidates 文件 → 跑通 ``combine`` 子命令 → 写出 combo spec。"""
    toks = _feasible(3)
    a = _write_candidates(tmp_path / "a.json", toks, n_trials=1200)
    b = _write_candidates(
        tmp_path / "b.json", _feasible(3, seed=5), n_trials=800, spec_id="b"
    )
    csv_path = _csv(tmp_path, n=1000)
    out = tmp_path / "combo.json"

    result = _runner.invoke(
        app,
        [
            "combine",
            "--candidates",
            f"{a},{b}",
            "--data",
            str(csv_path),
            "--method",
            "equal",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output

    # §5 要求的三值实测输出
    assert "verdict=" in result.output
    assert "DSR=" in result.output
    assert "n_trials=" in result.output
    # ComboPayload 自带 vocab，不得因「非 FactorPayload」就打成 "-"
    assert f"vocab={VOCAB_VERSION}" in result.output

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["payload"]["kind"] == "combo"
    assert payload["spec_version"] == "1.0"  # §1.6 不 bump 2.0
    assert payload["evidence"]["n_trials"] == 2000  # 1200 + 800，按运行去重
    assert payload["provenance"]["budget"] == "combo"
    assert payload["name"] == "combo_h1_xauusd"
    assert payload["semantics"]["timeframe"] == "H1"
    combo_meta = payload["provenance"]["config_snapshot"]["combo"]
    assert combo_meta["method"] == "equal"
    # a/b 是两组**不同**公式 → 各 3 条，跨文件无重复，共 6 个成员
    assert combo_meta["n_members"] == 6
    assert combo_meta["sources"] and len(combo_meta["sources"]) == 2
    # 往返：combo payload 能被 decode 回来
    restored = decode_spec(out.read_text(encoding="utf-8"))
    assert isinstance(restored.payload, ComboPayload)
    assert restored.payload.n_members == 6
    assert len(restored.payload.weights) == 6


def test_combine_cli_rejects_bad_method(tmp_path: Path) -> None:
    toks = _feasible(3)
    a = _write_candidates(tmp_path / "a.json", toks)
    b = _write_candidates(tmp_path / "b.json", _feasible(3, seed=5), spec_id="b")
    result = _runner.invoke(
        app,
        [
            "combine",
            "--candidates",
            f"{a},{b}",
            "--data",
            str(_csv(tmp_path)),
            "--method",
            "genetic",
            "--out",
            str(tmp_path / "c.json"),
        ],
    )
    assert result.exit_code == 1
    assert "未知权重方法" in result.output


def test_combine_cli_reports_missing_candidates(tmp_path: Path) -> None:
    result = _runner.invoke(
        app,
        [
            "combine",
            "--candidates",
            str(tmp_path / "nope.json"),
            "--data",
            str(_csv(tmp_path)),
            "--out",
            str(tmp_path / "c.json"),
        ],
    )
    assert result.exit_code == 1
    assert "组合失败" in result.output
    assert "候选文件不存在" in result.output
