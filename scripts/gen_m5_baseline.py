"""生成 M5 对拍的 **torch 基准**（运行在装有真实 torch 的解释器上）。

用 AM 的**原始 torch 实现**对确定性用例求值，产出：

* ``tests/fixtures/m5_baseline.npz`` —— 数值数组（``_normalize_output`` / signal 仓位 /
  StackVM 因子输出），加 ``tests/fixtures/m5_baseline.json`` 的元信息与标量结果
  （backtest 多目标分、evaluator 的 IC/RankIC/MI/prune/ablate/秩归一）。

妙算侧（``tests/parity/test_*_parity.py``）在**无 torch** 的 venv 中读取基准，用 numpy 实现
复算并比对。两侧解耦：先冻结输入，再对拍，从而排除「输入不同」导致误判。

运行（Oracle python，装 torch）：

    C:/Users/Administrator/.workbuddy/binaries/python/envs/default/Scripts/python.exe \\
        scripts/gen_m5_baseline.py

可用 ``MIAOSUAN_AM_ROOT`` 覆盖 AM 仓库位置。对 AM 仓库**零写入**。
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_AM_ROOT = Path(r"D:\backup\BaoBao\PythonProgram\AlphaMaster-main")
_FIX = _ROOT / "tests" / "fixtures"
_OUT_NPZ = _FIX / "m5_baseline.npz"
_OUT_JSON = _FIX / "m5_baseline.json"


def _am_root() -> Path:
    env = os.environ.get("MIAOSUAN_AM_ROOT")
    return Path(env) if env else _DEFAULT_AM_ROOT


def _dc(obj: Any) -> Any:
    """dataclass/list → JSON-ready（-inf → None）。"""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _dc(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _dc(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_dc(v) for v in obj]
    if isinstance(obj, float):
        if obj == float("inf") or obj == float("-inf") or obj != obj:
            return None
        return obj
    return obj


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 M5 对拍 torch 基准")
    parser.add_argument("--am-root", type=Path, default=None)
    parser.add_argument("--npz", type=Path, default=_OUT_NPZ)
    parser.add_argument("--json", type=Path, default=_OUT_JSON)
    args = parser.parse_args()

    am_root = args.am_root or _am_root()
    sys.path.insert(0, str(_ROOT / "tests" / "parity"))
    import m5_cases  # noqa: PLC0415

    import torch  # noqa: PLC0415

    prev = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(am_root))
    try:
        from model_core.backtest import MT5Backtest  # noqa: PLC0415
        from model_core.evaluator import (  # noqa: PLC0415
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
        from model_core.vm import StackVM  # noqa: PLC0415
        from strategy_manager.signal import compute_target_positions  # noqa: PLC0415
    finally:
        sys.path.remove(str(am_root))
        sys.dont_write_bytecode = prev

    npz: dict[str, np.ndarray] = {}
    meta: dict[str, Any] = {
        "_comment": "由 scripts/gen_m5_baseline.py 用冻结 AM 生成，请勿手工编辑。",
        "source": "AlphaMaster-main/model_core + strategy_manager/signal.py",
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "seed": m5_cases.SEED,
        "dtype": "float32",
    }

    def _t(a: np.ndarray) -> Any:
        return torch.from_numpy(np.ascontiguousarray(a.astype(np.float32, copy=False)))

    # ── 1) _normalize_output ─────────────────────────────────────────────
    norm_arrays = m5_cases.build_normalize_arrays()
    for name, arr in norm_arrays.items():
        npz[f"norm_in__{name}"] = arr.astype(np.float32, copy=False)
        with torch.no_grad():
            out = StackVM._normalize_output(_t(arr))
        npz[f"norm_out__{name}"] = out.detach().cpu().numpy().astype(np.float32, copy=False)
    meta["normalize_cases"] = sorted(norm_arrays)

    # ── 2) signal ────────────────────────────────────────────────────────
    sig_factors = m5_cases.build_signal_factors()
    for name, arr in sig_factors.items():
        npz[f"sig_in__{name}"] = arr.astype(np.float32, copy=False)
        with torch.no_grad():
            out = compute_target_positions(_t(arr))
        npz[f"sig_out__{name}"] = out.detach().cpu().numpy().astype(np.float32, copy=False)
    meta["signal_cases"] = sorted(sig_factors)

    # ── 3) StackVM.execute ───────────────────────────────────────────────
    vm = StackVM()
    panels = m5_cases.build_feat_panels()
    none_keys: list[str] = []
    for panel_name, panel in panels.items():
        npz[f"vmfeat__{panel_name}"] = panel.astype(np.float32, copy=False)
        for fname, formula in m5_cases.VM_FORMULAS.items():
            key = f"{panel_name}__{fname}"
            with torch.no_grad():
                res = vm.execute(list(formula), _t(panel))
            if res is None:
                none_keys.append(key)
            else:
                npz[f"vmout__{key}"] = res.detach().cpu().numpy().astype(np.float32, copy=False)
    meta["vm_none_keys"] = none_keys
    meta["vm_keys"] = [f"{p}__{f}" for p in panels for f in m5_cases.VM_FORMULAS]

    # ── 4) backtest ──────────────────────────────────────────────────────
    bt = MT5Backtest(cost_rate=0.0003, periods_per_year=6240)
    bt_results: dict[str, Any] = {}
    bt_cases = m5_cases.build_backtest_cases()
    for name, payload in bt_cases.items():
        factors = _t(payload["factors"])
        target = _t(payload["target_ret"])
        with torch.no_grad():
            sc, mean_oos = bt.evaluate(factors, {}, target)
            t_full = payload["factors"].shape[1]
            split = int(t_full * 0.6)
            tr, vl = bt.evaluate_fold(factors, target, 0, split, split, t_full)
        bt_results[name] = {
            "score": float(sc.item() if hasattr(sc, "item") else sc),
            "mean_oos": float(mean_oos),
            "fold_split": split,
            "fold_train": float(tr.item() if hasattr(tr, "item") else tr),
            "fold_val": float(vl.item() if hasattr(vl, "item") else vl),
        }
        npz[f"bt_factors__{name}"] = payload["factors"].astype(np.float32, copy=False)
        npz[f"bt_target__{name}"] = payload["target_ret"].astype(np.float32, copy=False)
    meta["backtest"] = bt_results

    # ── 5) evaluator ─────────────────────────────────────────────────────
    ev_cases = m5_cases.build_evaluator_cases()
    cands: dict[str, np.ndarray] = ev_cases["candidates"]
    target = ev_cases["target"]
    horizon = int(ev_cases["horizon"])
    for cname, c in cands.items():
        npz[f"ev_cand__{cname}"] = c.astype(np.float32, copy=False)
    npz["ev_target"] = target.astype(np.float32, copy=False)

    ev_results: dict[str, Any] = {}
    # 单候选 score（逐个），用 horizon
    single: dict[str, Any] = {}
    for cname, c in cands.items():
        with torch.no_grad():
            sr = score(_t(c), _t(target), name=cname, category=cname, horizon=horizon)
        single[cname] = _dc(sr)
    ev_results["score_single"] = single

    cands_t = {k: _t(v) for k, v in cands.items()}
    target_t = _t(target)
    with torch.no_grad():
        all_sr = score_all(cands_t, target_t, categories=ev_cases["categories"], horizon=horizon)
    ev_results["score_all"] = _dc(all_sr)

    with torch.no_grad():
        rows = prune(all_sr, cands_t, corr_threshold=0.9, conservative=False, margin=0.01)
    ev_results["prune"] = _dc(rows)

    with torch.no_grad():
        abl = ablate("A", cands_t, target_t, drop_threshold=-0.01, horizon=horizon)
    ev_results["ablate_A"] = _dc(abl)
    with torch.no_grad():
        abl_degen = ablate("DEGEN", cands_t, target_t, drop_threshold=-0.01, horizon=horizon)
    ev_results["ablate_DEGEN"] = _dc(abl_degen)

    # 秩归一
    raw_vals = [0.1, -0.5, 0.3, -float("inf"), 0.9, 0.0]
    ev_results["rank_normalize_in"] = raw_vals
    ev_results["rank_normalize_out"] = _rank_normalize(list(raw_vals))

    # 底层函数
    a = cands["A"]
    b = cands["D"]
    with torch.no_grad():
        ic, ric, ir = _compute_ic_rankic(_t(a), _t(target))
        mi = _compute_mi(_t(a), _t(target))
        degen_true = _is_degenerate(_t(cands["DEGEN"]))
        degen_false = _is_degenerate(_t(a))
        corr = _pearson_corr(_t(a), _t(b))
        ca, ta = _align_causal(_t(a), _t(target), horizon)
    ev_results["ic_rankic_A"] = {"ic": ic, "rank_ic": ric, "ir": ir}
    ev_results["mi_A"] = mi
    ev_results["is_degenerate"] = {"DEGEN": degen_true, "A": degen_false}
    ev_results["pearson_corr_AD"] = corr
    ev_results["align_causal_shape"] = list(ca.shape)
    npz["ev_align_cand_A"] = ca.detach().cpu().numpy().astype(np.float32, copy=False)
    npz["ev_align_target_A"] = ta.detach().cpu().numpy().astype(np.float32, copy=False)

    # report + active subset（generated_at 冻结以便对拍）
    report = build_report(
        rows, active_subset=[], vocab_version="v9217a2c0d91a", config={}
    )
    ev_results["report_row_order"] = [r.candidate for r in report.rows]
    ev_results["report_retention"] = [r.retention_status for r in report.rows]
    subset = select_active_subset(report, retention_threshold=0.0, max_retained=None)
    ev_results["active_subset"] = subset
    ev_results["horizon"] = horizon

    meta["evaluator"] = ev_results
    meta["evaluator_policy"] = {
        "corr_threshold": 0.9,
        "conservative": False,
        "margin": 0.01,
        "drop_threshold": -0.01,
        "horizon": horizon,
    }

    meta["npz_keys"] = sorted(npz)

    args.npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.npz, **npz)
    with args.json.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    print(f"[OK] AM 根: {am_root}")
    print(f"[OK] torch={torch.__version__} numpy={np.__version__}")
    print(f"[OK] npz keys={len(npz)}; vm none={len(none_keys)}")
    print(f"[OK] 写出: {args.npz}")
    print(f"[OK] 写出: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
