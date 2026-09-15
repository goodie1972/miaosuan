# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""M5 评估器数值对拍：妙算 numpy 实现 vs 冻结 AM 的 torch 基准。

覆盖（team-lead M5 难点 4 —— IC/RankIC/Score/退化/``_align_causal`` **原样保留**）：

* 底层函数：``_align_causal`` / ``_compute_ic_rankic`` / ``_compute_mi`` / ``_is_degenerate``
  / ``_pearson_corr`` / ``_rank_normalize``；
* 打分：``score``（单候选） / ``score_all``（跨候选秩归一）；
* 剪枝：``prune``（保守双条件）；
* 消融：``ablate``（含退化候选）；
* 报告：``build_report``（排序 + 冻结时间戳） / ``select_active_subset``；
* 类封装：``EffectivenessEvaluator`` 与模块级函数一致；
* ``report.persist`` 的 IO 往返（新搬迁的 IO 半边）。

基准由 ``scripts/gen_m5_baseline.py`` 在真实 torch 环境用 AM 原始实现生成
（``generated_at`` 已在基准侧冻结，便于对拍）。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import m5_cases  # noqa: E402

from miaosuan.core.evaluator import (  # noqa: E402
    EffectivenessEvaluator,
    _align_causal,
    _compute_ic_rankic,
    _compute_mi,
    _is_degenerate,
    _pearson_corr,
    _rank_normalize,
    ablate,
    build_report,
    prune,
    score,
    score_all,
    select_active_subset,
)
from miaosuan.report.persist import (  # noqa: E402
    load_report,
    report_from_dict,
    report_to_dict,
    save_report,
)

pytestmark = pytest.mark.parity

# 评估器全程 float64 打分；仅输入张量为 float32。实测 Δ ≤ 1e-8，容差留 3 个数量级。
_SC_ATOL = 1e-6
_SC_RTOL = 1e-4


def _close(got: float, exp: float, atol: float = _SC_ATOL, rtol: float = _SC_RTOL) -> bool:
    return abs(got - exp) <= atol + rtol * max(1.0, abs(exp))


def _cases() -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, str], int]:
    ev = m5_cases.build_evaluator_cases()
    return ev["candidates"], ev["target"], ev["categories"], int(ev["horizon"])


def _npz_cases(m5_npz: dict) -> tuple[dict[str, np.ndarray], np.ndarray, int]:
    """从基准 npz 读取输入（与生成侧逐位一致），避免测试侧重算 RNG 漂移。"""
    cands = {
        name: m5_npz[f"ev_cand__{name}"]
        for name in ("A", "B", "C", "D", "DEGEN")
    }
    return cands, m5_npz["ev_target"], 2


# ── 底层函数对拍 ────────────────────────────────────────────────────────────


def test_input_consistency(m5_npz: dict) -> None:
    """测试侧构造的输入必须与基准输入逐位一致（排除 RNG 漂移）。"""
    cands, target, _cats, _h = _cases()
    for name, c in cands.items():
        assert np.array_equal(c, m5_npz[f"ev_cand__{name}"]), f"候选 {name} 输入不一致"
    assert np.array_equal(target, m5_npz["ev_target"]), "target 输入不一致"


def test_align_causal_parity(m5_npz: dict, m5_meta: dict) -> None:
    """``_align_causal`` 原样保留：末端裁掉 horizon 步，形状与数值逐点一致。"""
    ev = m5_meta["evaluator"]
    cands, target, _cats, horizon = _cases()
    ca, ta = _align_causal(cands["A"], target, horizon)
    assert list(ca.shape) == ev["align_causal_shape"] == [3, 398]
    assert list(ta.shape) == ev["align_causal_shape"]
    assert np.allclose(ca, m5_npz["ev_align_cand_A"], atol=0.0, rtol=0.0)
    assert np.allclose(ta, m5_npz["ev_align_target_A"], atol=0.0, rtol=0.0)
    # 防 look-ahead 语义：只保留前 T-h 列
    assert np.array_equal(ca, cands["A"][..., : target.shape[-1] - horizon])


def test_ic_rankic_mi_degenerate_pearson(m5_npz: dict, m5_meta: dict) -> None:
    ev = m5_meta["evaluator"]
    cands, target, _cats, _h = _cases()

    ic, ric, ir = _compute_ic_rankic(cands["A"], target)
    assert _close(ic, ev["ic_rankic_A"]["ic"])
    assert _close(ric, ev["ic_rankic_A"]["rank_ic"])
    assert _close(ir, ev["ic_rankic_A"]["ir"], rtol=1e-3)

    assert _close(_compute_mi(cands["A"], target), ev["mi_A"])
    assert _is_degenerate(cands["DEGEN"]) is ev["is_degenerate"]["DEGEN"]
    assert _is_degenerate(cands["A"]) is ev["is_degenerate"]["A"]
    assert _close(_pearson_corr(cands["A"], cands["D"]), ev["pearson_corr_AD"])


def test_rank_normalize_parity(m5_meta: dict) -> None:
    ev = m5_meta["evaluator"]
    got = _rank_normalize(list(ev["rank_normalize_in"]))
    exp = ev["rank_normalize_out"]
    assert len(got) == len(exp)
    for g, e in zip(got, exp, strict=True):
        assert _close(g, e, atol=1e-12, rtol=0.0), f"秩归一 {g} != {e}"


# ── 打分 / 剪枝 / 消融 对拍 ────────────────────────────────────────────────


def test_score_single_parity(m5_npz: dict, m5_meta: dict) -> None:
    ev = m5_meta["evaluator"]["score_single"]
    cands, target, _cats, horizon = _cases()
    for name, exp in ev.items():
        sr = score(cands[name], target, name=name, category=name, horizon=horizon)
        assert sr.candidate == exp["candidate"]
        assert sr.category == exp["category"]
        assert sr.unscorable is exp["unscorable"]
        assert _close(sr.ic, exp["ic"])
        assert _close(sr.rank_ic, exp["rank_ic"])
        assert _close(sr.ir, exp["ir"], rtol=1e-3)
        assert _close(sr.mi, exp["mi"])
        if exp["importance_score"] is None:
            assert not math.isfinite(sr.importance_score)
        else:
            assert _close(sr.importance_score, exp["importance_score"])


def test_score_all_parity(m5_npz: dict, m5_meta: dict) -> None:
    exp_all = m5_meta["evaluator"]["score_all"]
    cands, target, cats, horizon = _cases()
    got = score_all(cands, target, categories=cats, horizon=horizon)
    assert [r.candidate for r in got] == [e["candidate"] for e in exp_all]
    for r, e in zip(got, exp_all, strict=True):
        assert r.category == e["category"]
        assert r.unscorable is e["unscorable"]
        assert _close(r.rank_ic, e["rank_ic"])
        assert _close(r.mi, e["mi"])
        if e["importance_score"] is None:
            assert not math.isfinite(r.importance_score)
        else:
            assert _close(r.importance_score, e["importance_score"])


def test_prune_parity(m5_npz: dict, m5_meta: dict) -> None:
    ev = m5_meta["evaluator"]
    cands, target, cats, horizon = _cases()
    all_sr = score_all(cands, target, categories=cats, horizon=horizon)
    rows = prune(all_sr, cands, corr_threshold=0.9, conservative=False, margin=0.01)
    exp = ev["prune"]
    assert [r.candidate for r in rows] == [e["candidate"] for e in exp]
    for r, e in zip(rows, exp, strict=True):
        assert r.retention_status == e["retention_status"], f"{r.candidate} 保留状态不一致"
        assert r.pruned_in_favor_of == e["pruned_in_favor_of"], f"{r.candidate} 剪枝去向不一致"


def test_ablate_parity(m5_npz: dict, m5_meta: dict) -> None:
    ev = m5_meta["evaluator"]
    cands, target, _cats, horizon = _cases()
    a_exp = ev["ablate_A"]
    got_a = ablate("A", cands, target, drop_threshold=-0.01, horizon=horizon)
    assert got_a.name == a_exp["name"]
    assert got_a.error == a_exp["error"]
    assert _close(got_a.marginal_contribution, a_exp["marginal_contribution"])
    assert got_a.drop_recommendation is a_exp["drop_recommendation"]

    d_exp = ev["ablate_DEGEN"]
    got_d = ablate("DEGEN", cands, target, drop_threshold=-0.01, horizon=horizon)
    assert got_d.error == d_exp["error"]
    assert _close(got_d.marginal_contribution, d_exp["marginal_contribution"])
    assert got_d.drop_recommendation is d_exp["drop_recommendation"]


def test_ablate_missing_name_returns_error() -> None:
    """消融不存在的候选 → 返回 error 而非抛异常（与 AM 一致）。"""
    cands, target, _cats, horizon = _cases()
    res = ablate("NOT_IN_SET", cands, target, horizon=horizon)
    assert res.error is not None
    assert res.marginal_contribution is None
    assert res.drop_recommendation is False


# ── 报告 / Active_Subset 对拍 ──────────────────────────────────────────────


def test_report_and_active_subset_parity(m5_npz: dict, m5_meta: dict) -> None:
    ev = m5_meta["evaluator"]
    cands, target, cats, horizon = _cases()
    all_sr = score_all(cands, target, categories=cats, horizon=horizon)
    rows = prune(all_sr, cands, corr_threshold=0.9, conservative=False, margin=0.01)

    report = build_report(rows, active_subset=[], vocab_version="v9217a2c0d91a", config={})
    assert [r.candidate for r in report.rows] == ev["report_row_order"]
    assert [r.retention_status for r in report.rows] == ev["report_retention"]

    subset = select_active_subset(report, retention_threshold=0.0, max_retained=None)
    assert subset == ev["active_subset"] == ["D", "B", "DEGEN"]


def test_build_report_generated_at_injectable() -> None:
    """``generated_at`` 可注入（可复现），默认取 UTC now。"""
    frozen = "2026-09-10T00:00:00+00:00"
    r = build_report([], active_subset=[], vocab_version="vX", config={}, generated_at=frozen)
    assert r.generated_at == frozen
    r2 = build_report([], active_subset=[], vocab_version="vX", config={})
    assert r2.generated_at and r2.generated_at != frozen


def test_report_row_order_deterministic() -> None:
    """排序 tie-break 为名称字母序（确定性）。"""
    cands, target, cats, horizon = _cases()
    rows = prune(score_all(cands, target, categories=cats, horizon=horizon), cands)
    report = build_report(rows, active_subset=[], vocab_version="vX", config={})
    scores = [r.importance_score for r in report.rows]
    finite = [s for s in scores if math.isfinite(s)]
    assert finite == sorted(finite, reverse=True), "报告未按 importance_score 降序"


# ── 类封装一致性 ──────────────────────────────────────────────────────────


def test_evaluator_class_matches_functions(m5_npz: dict, m5_meta: dict) -> None:
    """``EffectivenessEvaluator`` 默认参数下与模块级函数逐点一致。"""
    ev = m5_meta["evaluator"]
    cands, target, cats, _h = _cases()
    eva = EffectivenessEvaluator()  # target_horizon=2 默认，与基准一致

    all_sr = eva.score_all(cands, target, categories=cats)
    rows = eva.prune(all_sr, cands)
    assert [r.candidate for r in rows] == [e["candidate"] for e in ev["prune"]]

    report = eva.build_report(rows, [], generated_at="2026-09-10T00:00:00+00:00")
    assert [r.candidate for r in report.rows] == ev["report_row_order"]
    assert eva.select_active_subset(report) == ev["active_subset"]

    assert _close(eva.score(cands["A"], target, "A", "trend").ic, ev["score_single"]["A"]["ic"])


# ── report.persist IO 往返 ────────────────────────────────────────────────


def test_report_persist_round_trip(tmp_path: Path, m5_npz: dict, m5_meta: dict) -> None:
    """``report/persist`` 的 dict 往返与文件读写（原子写）保真。"""
    ev = m5_meta["evaluator"]
    cands, target, cats, horizon = _cases()
    rows = prune(score_all(cands, target, categories=cats, horizon=horizon), cands)
    report = build_report(
        rows, active_subset=ev["active_subset"], vocab_version="v9217a2c0d91a", config={},
        generated_at="2026-09-10T00:00:00+00:00",
    )

    d = report_to_dict(report)
    assert d["vocab_version"] == "v9217a2c0d91a"
    restored = report_from_dict(d)
    assert [r.candidate for r in restored.rows] == [r.candidate for r in report.rows]
    assert restored.active_subset == report.active_subset
    # -inf（unscorable）经 JSON 往返后仍为 -inf 哨兵
    degen = next(r for r in restored.rows if r.candidate == "DEGEN")
    assert degen.unscorable and not math.isfinite(degen.importance_score)

    path = tmp_path / "report.json"
    save_report(report, str(path))
    assert path.is_file()
    loaded = load_report(str(path))
    assert [r.candidate for r in loaded.rows] == [r.candidate for r in report.rows]
    assert loaded.vocab_version == report.vocab_version
