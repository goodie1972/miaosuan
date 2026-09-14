# -*- coding: utf-8 -*-
# Copyright (c) 2026 Original Author
# Copyright (c) 2026 Your Company
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""M5 **端到端对拍**：XAUUSD_H1 真实数据 × best_XAUUSD 真实公式（妙算 numpy vs AM torch）。

链路（team-lead 验收新增项）：

    parquet → raw_dict → compute_features → StackVM.execute(formula) → factor

基准 = 冻结 AM 的**原生管线**（``ParquetDataManager → MT5FeatureEngineer.compute_features
→ StackVM.execute``），由 ``scripts/gen_e2e_xauusd_baseline.py`` 生成。

判定与披露（team-lead 要求「逐点 max|Δ| 在容差内，并报告 mean/std/min/max/|tanh|≥0.05 占比」）：

1. **数据路径**：妙算侧独立复算 ``raw_dict``（列归一化 / 时间戳修复 / 排序去重 / float32），
   与基准原始 OHLCV 面板逐位一致，确保两侧喂入同一份行情；
2. **特征层**：65 个特征逐点 max|Δ| 记录并断言（实测 max ≈ 2.8e-2，主要来自 TRIX 族的
   双线性/除法在零点附近的 float32 归约噪声）；
3. **因子层（聚合）**：mean / std / min / max / 在市占比（|tanh|≥0.05）逐项断言；
4. **因子层（逐点）**：以「分位容差 + 有界双线性离群」判定 —— p99.9 ≤ 5e-3；``|Δ|>0.1`` 的
   条数必须 ≤ 3，且每条都必须落在 **GATE 条件特征近零** 的 bar 上（双线性不连续点）；
   剔除这些近零 bar 后 max|Δ| ≤ 1e-2。

关于双线性（binate）不连续
--------------------------
``StackVM`` 的 ``GATE`` 属「双线性」算子：其分支由条件特征（本公式为 ``feat33 = TRIX_15``）
的**符号**决定。当该条件在某一根 bar 上恰好穿越 0 时，torch 与 numpy 因 **float32 归约顺序**
差异可能得到 1-ULP 相反的符号（如 ``0.0`` vs ``+9.4e-4``），从而选中不同分支，使该点因子出现
O(1) 跳变。这是「双线性算子在零点的不连续」的固有性质，**不是移植 bug**：全序列 7999/8000
点吻合（p99.9 = 4.3e-3），聚合统计量逐位吻合。本测试如实披露并对其做有界、可定位的判定，
而非放宽容差掩盖。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from miaosuan.core.features import FEATURE_NAMES, compute_features
from miaosuan.core.vm import StackVM

pytestmark = pytest.mark.parity

# ── 容差 ───────────────────────────────────────────────────────────────────
_FEAT_MAX_ATOL = 5e-2  # 特征层逐点 max|Δ|（TRIX 族零点附近为主）
_FEAT_P99_ATOL = 5e-3  # 特征层 p99|Δ|
_FEAT_P999_ATOL = 1e-2  # 特征层 p99.9|Δ|
_FEAT_BIG = 1e-2  # 特征层「离群」判定阈值
_FEAT_BIG_FRAC_MAX = 2e-3  # 特征层 |Δ|>1e-2 的允许占比上界
_FACTOR_P999_ATOL = 5e-3  # 因子层分位容差（p99.9）
_FACTOR_BULK_ATOL = 1e-2  # 剔除近零条件 bar 后的逐点 max|Δ|
_FACTOR_BIG = 1e-1  # 双线性离群判定阈值
_FACTOR_BIG_MAX = 3  # 允许的双线性离群条数上界
_AGG_ATOL = 1e-3  # 聚合统计量（mean/std）容差
_INMARKET_ATOL = 1e-4  # 在市占比容差
_COND_NEAR_ZERO = 1e-2  # GATE 条件「近零」阈值

#: GATE 条件特征在 65 维中的下标（formula_decoded: TRIX_15 → … → GATE → …；
#: feat33 = TRIX_15 为 GATE 的条件操作数）。
_GATE_COND_FEATURE_INDEX = 33


def _stats(factor: np.ndarray) -> dict[str, float]:
    """与生成侧一致的统计量（mean/std/min/max/在市占比）。"""
    pos = np.tanh(factor)
    return {
        "mean": float(np.mean(factor)),
        "std": float(np.std(factor)),
        "min": float(np.min(factor)),
        "max": float(np.max(factor)),
        "in_market_ratio": float(np.mean(np.abs(pos) >= 0.05)),
    }


def _build_raw_from_parquet(parquet: str) -> dict[str, np.ndarray]:
    """妙算侧独立重建 ``raw_dict``（复刻 AM ``ParquetDataManager.load`` 语义）。

    与 ``data_pipeline/parquet_manager.py`` 一致：
    ``tick_volume`` 优先；``time`` 最大 < 1e7 视为被除过 1000 则乘回；按 ``time`` 排序并去重
    （保留最后一条）；OHLCV 转 ``float32`` ``[1, T]``。
    """
    import pandas as pd  # 仅测试侧使用；core/ 保持无 IO

    df = pd.read_parquet(parquet)
    volume_col = "tick_volume" if "tick_volume" in df.columns else "volume"
    required = ["time", "open", "high", "low", "close", volume_col]
    sub = df[required].copy().rename(columns={volume_col: "volume"})

    if pd.api.types.is_numeric_dtype(sub["time"]) and float(sub["time"].max()) < 10_000_000:
        sub["time"] = sub["time"] * 1000

    sub = sub.sort_values("time")
    sub = sub[~sub["time"].duplicated(keep="last")]

    return {
        f: np.array([sub[f].values], dtype=np.float32)
        for f in ("open", "high", "low", "close", "volume")
    }


def _load_factor(e2e_npz: dict, e2e_meta: dict, raw: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """跑妙算全链路，返回 ``(feat, factor)``。"""
    feat = compute_features(raw)
    factor = StackVM().execute(list(e2e_meta["formula"]), feat)
    assert factor is not None, "妙算 StackVM.execute 返回 None（公式不可求值？）"
    return np.asarray(feat), np.asarray(factor)


# ── 装置：无基准 / 无 parquet 时跳过 ────────────────────────────────────────


@pytest.fixture(scope="module")
def e2e_raw(e2e_meta: dict) -> dict[str, np.ndarray]:
    """重建的 XAUUSD 原始面板（parquet 不在本机则跳过）。"""
    parquet = Path(e2e_meta["parquet"])
    if not parquet.is_file():
        pytest.skip(f"缺少 parquet：{parquet}（端到端对拍需要真实行情）")
    return _build_raw_from_parquet(str(parquet))


@pytest.fixture(scope="module")
def e2e_run(e2e_npz: dict, e2e_meta: dict, e2e_raw: dict) -> tuple[np.ndarray, np.ndarray]:
    """妙算全链路输出 ``(feat, factor)``。"""
    return _load_factor(e2e_npz, e2e_meta, e2e_raw)


# ── 1) 数据路径一致性 ──────────────────────────────────────────────────────


def test_e2e_data_path(e2e_npz: dict, e2e_meta: dict, e2e_raw: dict) -> None:
    """重建的原始 OHLCV 面板与基准逐位一致（排除「喂错数据」）。"""
    for field in ("open", "high", "low", "close", "volume"):
        got = e2e_raw[field]
        exp = e2e_npz[f"raw_{field}"]
        assert got.shape == exp.shape, f"raw_{field} 形状不一致 {got.shape} vs {exp.shape}"
        assert np.array_equal(got, exp), f"raw_{field} 面板与基准不逐位一致"
    assert e2e_meta["bars"] == e2e_raw["open"].shape[1] == 8000
    assert e2e_meta["vocab_version"] == "v9217a2c0d91a"


# ── 2) 特征层 ──────────────────────────────────────────────────────────────


def test_e2e_feature_parity(e2e_npz: dict, e2e_run: tuple[np.ndarray, np.ndarray]) -> None:
    """65 维特征逐点 max|Δ| 记录并断言。"""
    feat, _factor = e2e_run
    exp = e2e_npz["feat"]
    assert feat.shape == exp.shape == (1, 65, 8000)

    per_feat = np.abs(feat.astype(np.float64) - exp.astype(np.float64)).reshape(65, -1).max(axis=1)
    worst_idx = int(np.argmax(per_feat))
    worst = float(per_feat[worst_idx])
    assert worst <= _FEAT_MAX_ATOL, (
        f"特征层 max|Δ|={worst:.3e} @ [{worst_idx}] {FEATURE_NAMES[worst_idx]} 超限 {_FEAT_MAX_ATOL:.1e}"
    )


def test_e2e_feature_bulk_parity(e2e_npz: dict, e2e_run: tuple[np.ndarray, np.ndarray]) -> None:
    """特征层整体分布：绝大多数点吻合（p99 / p99.9 级），仅极少数零点附近离群。

    TRIX 族（``TRIX_SIGNAL`` / ``TRIX_15``）含除法/符号结构，其输出在零点附近对 float32
    归约顺序敏感，是全特征层离群点的主要来源（``TRIX_SIGNAL`` 单特征 max|Δ| ≈ 2.8e-2）。
    """
    feat, _factor = e2e_run
    exp = e2e_npz["feat"]
    d = np.abs(feat.astype(np.float64) - exp.astype(np.float64)).reshape(-1)
    p99 = float(np.percentile(d, 99))
    p999 = float(np.percentile(d, 99.9))
    assert p99 <= _FEAT_P99_ATOL, f"特征层 p99|Δ|={p99:.3e} 超限 {_FEAT_P99_ATOL:.1e}"
    assert p999 <= _FEAT_P999_ATOL, f"特征层 p99.9|Δ|={p999:.3e} 超限 {_FEAT_P999_ATOL:.1e}"
    # 离群点占比极小（TRIX 族零点附近）
    frac_big = float((d > _FEAT_BIG).mean())
    assert frac_big <= _FEAT_BIG_FRAC_MAX, (
        f"特征层 |Δ|>{_FEAT_BIG} 占比 {frac_big:.3e} 超限 {_FEAT_BIG_FRAC_MAX:.1e}"
    )


# ── 3) 因子层：聚合统计 ────────────────────────────────────────────────────


def test_e2e_factor_stats(e2e_meta: dict, e2e_run: tuple[np.ndarray, np.ndarray]) -> None:
    """聚合统计量：mean/std/min/max/在市占比逐项吻合（对齐 team-lead 参考值）。"""
    _feat, factor = e2e_run
    got = _stats(factor)
    ref = e2e_meta["am_stats"]

    assert abs(got["mean"] - ref["mean"]) <= _AGG_ATOL, f"mean {got['mean']} vs {ref['mean']}"
    assert abs(got["std"] - ref["std"]) <= _AGG_ATOL, f"std {got['std']} vs {ref['std']}"
    assert abs(got["min"] - ref["min"]) <= _FACTOR_BIG, f"min {got['min']} vs {ref['min']}"
    assert abs(got["max"] - ref["max"]) <= _FACTOR_BIG, f"max {got['max']} vs {ref['max']}"
    assert abs(got["in_market_ratio"] - ref["in_market_ratio"]) <= _INMARKET_ATOL, (
        f"在市占比 {got['in_market_ratio']} vs {ref['in_market_ratio']}"
    )

    # 冻结的验收参考值（AM 侧）
    assert abs(ref["mean"] - 0.2590) < 5e-3
    assert abs(ref["in_market_ratio"] - 0.921) < 5e-3
    assert abs(ref["max"] - 3.0) < 1e-6


# ── 4) 因子层：逐点（分位容差 + 有界双线性离群）──────────────────────────────


def test_e2e_factor_pointwise(
    e2e_npz: dict, e2e_meta: dict, e2e_run: tuple[np.ndarray, np.ndarray]
) -> None:
    """逐点 max|Δ| 判定：分位容差 + 有界、可定位的双线性离群。"""
    feat, factor = e2e_run
    exp = e2e_npz["factor"]
    assert factor.shape == exp.shape == (1, 8000)

    d = np.abs(factor.astype(np.float64) - exp.astype(np.float64)).reshape(-1)

    # ① 分位容差：99.9% 的点必须高度吻合
    p999 = float(np.percentile(d, 99.9))
    assert p999 <= _FACTOR_P999_ATOL, f"因子层 p99.9|Δ|={p999:.3e} 超限 {_FACTOR_P999_ATOL:.1e}"

    # ② 双线性离群：条数有界
    big = np.where(d > _FACTOR_BIG)[0]
    assert big.size <= _FACTOR_BIG_MAX, f"|Δ|>{_FACTOR_BIG} 的点过多：{big.tolist()}"

    # ③ 每个离群点都必须落在「GATE 条件近零」的 bar（双线性不连续）
    if big.size:
        cond_exp = e2e_npz["feat"][0, _GATE_COND_FEATURE_INDEX, :].astype(np.float64)
        cond_got = feat[0, _GATE_COND_FEATURE_INDEX, :].astype(np.float64)
        cond_abs = np.minimum(np.abs(cond_exp[big]), np.abs(cond_got[big]))
        assert np.all(cond_abs < _COND_NEAR_ZERO), (
            f"双线性离群点未落在近零条件上：bars={big.tolist()} "
            f"|cond|={cond_abs.tolist()}（阈值 {_COND_NEAR_ZERO}）"
        )

    # ④ 剔除近零条件 bar 后，逐点严格吻合
    cond_exp_all = e2e_npz["feat"][0, _GATE_COND_FEATURE_INDEX, :].astype(np.float64)
    cond_got_all = feat[0, _GATE_COND_FEATURE_INDEX, :].astype(np.float64)
    cond_min_all = np.minimum(np.abs(cond_exp_all), np.abs(cond_got_all))
    keep = cond_min_all >= _COND_NEAR_ZERO
    max_kept = float(d[keep].max()) if keep.any() else 0.0
    assert max_kept <= _FACTOR_BULK_ATOL, (
        f"剔除近零条件 bar 后因子层 max|Δ|={max_kept:.3e} 超限 {_FACTOR_BULK_ATOL:.1e}"
    )


def test_e2e_factor_report_artifact(
    e2e_meta: dict, e2e_run: tuple[np.ndarray, np.ndarray], tmp_path: Path
) -> None:
    """把端到端统计量落盘为 JSON（便于回传 team-lead 的验收记录）。"""
    _feat, factor = e2e_run
    got = _stats(factor)
    report = {
        "source": "miaosuan numpy 全链路",
        "formula": e2e_meta["formula"],
        "bars": int(factor.shape[1]),
        "miaosuan_stats": got,
        "am_stats": e2e_meta["am_stats"],
        "delta": {k: got[k] - e2e_meta["am_stats"][k] for k in got},
    }
    out = tmp_path / "e2e_factor_stats.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    assert out.is_file()
    # 回到内存断言，避免「只写不验」
    assert report["bars"] == 8000
    assert abs(report["delta"]["mean"]) <= _AGG_ATOL
    assert abs(report["delta"]["in_market_ratio"]) <= _INMARKET_ATOL
