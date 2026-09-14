# -*- coding: utf-8 -*-
# Copyright (c) 2026 Original Author
# Copyright (c) 2026 Your Company
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""生成 M4 特征对拍的 **torch 基准**（运行在装有真实 torch 的解释器上）。

用 AM 的**原始 torch 实现**（``model_core.features.MT5FeatureEngineer.compute_features``）
对确定性 OHLCV 用例求值，产出：

* ``tests/fixtures/frozen_feature_inputs.npz`` —— 冻结的输入面板（每个 case 的
  ``close/high/low/open/volume``），加 ``tests/fixtures/feature_inputs_meta.json``；
* ``tests/fixtures/features_baseline.npz`` —— AM 原实现的输出 ``[N, F, T]``（每 case），
  加 ``tests/fixtures/features_baseline_meta.json``。

妙算侧（``tests/parity/test_feature_parity.py``）在**无 torch** 的 venv 中读取基准，
用 numpy 实现复算并比对 ``max|Δ|``。两侧解耦：先冻结输入，再对拍，从而排除
「输入不同」导致误判。

运行：

    C:\Users\Administrator\.workbuddy\binaries\python\envs\default\Scripts\python.exe \
        scripts/gen_feature_baseline.py

可用 ``MIAOSUAN_AM_ROOT`` 覆盖 AM 仓库位置。对 AM 仓库**零写入**。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_AM_ROOT = Path(r"D:ackup\BaoBao\PythonProgram\AlphaMaster-main")
_FIX = _ROOT / "tests" / "fixtures"
_OUT_INPUTS = _FIX / "frozen_feature_inputs.npz"
_OUT_INPUTS_META = _FIX / "feature_inputs_meta.json"
_OUT_BASELINE = _FIX / "features_baseline.npz"
_OUT_BASELINE_META = _FIX / "features_baseline_meta.json"


def _am_root() -> Path:
    env = os.environ.get("MIAOSUAN_AM_ROOT")
    return Path(env) if env else _DEFAULT_AM_ROOT


def main() -> int:
    parser = argparse.ArgumentParser(description="生成特征对拍 torch 基准")
    parser.add_argument("--am-root", type=Path, default=None)
    parser.add_argument("--inputs", type=Path, default=_OUT_INPUTS)
    parser.add_argument("--inputs-meta", type=Path, default=_OUT_INPUTS_META)
    parser.add_argument("--baseline", type=Path, default=_OUT_BASELINE)
    parser.add_argument("--baseline-meta", type=Path, default=_OUT_BASELINE_META)
    args = parser.parse_args()

    am_root = args.am_root or _am_root()

    # 共享用例（纯 numpy）
    sys.path.insert(0, str(_ROOT / "tests" / "parity"))
    import feature_cases  # noqa: PLC0415

    import torch  # noqa: PLC0415

    prev = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(am_root))
    try:
        from model_core.features import FEATURE_REGISTRY, MT5FeatureEngineer  # noqa: PLC0415
    finally:
        sys.path.remove(str(am_root))
        sys.dont_write_bytecode = prev

    feature_names = [spec.name for spec in FEATURE_REGISTRY.feature_specs]
    if len(feature_names) != 65:
        raise SystemExit(
            f"[ERROR] AM 注册特征数 {len(feature_names)} != 65（疑似 active_features.json 生效）"
        )

    cases = feature_cases.build_cases()
    inputs: dict[str, np.ndarray] = {}
    outputs: dict[str, np.ndarray] = {}
    helpers: dict[str, np.ndarray] = {}
    case_meta: list[dict] = []

    for case_name, payload in cases.items():
        raw_t = {}
        for field in feature_cases.FIELDS:
            arr = np.ascontiguousarray(payload[field].astype(np.float32, copy=False))
            inputs[f"in_{case_name}_{field}"] = arr
            raw_t[field] = torch.from_numpy(arr)
        close, high, low = raw_t["close"], raw_t["high"], raw_t["low"]
        with torch.no_grad():
            out = MT5FeatureEngineer.compute_features(raw_t)
            # 共享底层 helper 输出（紧容差对拍用，隔离归一化放大）
            h = {
                "ema_span12": MT5FeatureEngineer._ema_simple(close, 12),
                "ema_span15": MT5FeatureEngineer._ema_simple(close, 15),
                "ema_span20": MT5FeatureEngineer._ema_simple(close, 20),
                "ema_span26": MT5FeatureEngineer._ema_simple(close, 26),
                "ema_span50": MT5FeatureEngineer._ema_simple(close, 50),
                "ma10": MT5FeatureEngineer._ma(close, 10),
                "rolling_std20": MT5FeatureEngineer._rolling_std(close, 20),
                "atr14": MT5FeatureEngineer._atr(close, high, low, 14),
                "rvol": MT5FeatureEngineer._rvol(close),
                "ac1": MT5FeatureEngineer._ac1(close),
                "linear_slope20": MT5FeatureEngineer._linear_slope(close, 20),
                "trend_strength50": MT5FeatureEngineer._trend_strength(close, 50),
                "ts_corr10": MT5FeatureEngineer._ts_corr(close, close, 10),
                "robust_norm_close": MT5FeatureEngineer._robust_norm(close, 200),
                "trix15": MT5FeatureEngineer._trix(close, 15),
            }
        for key, val in h.items():
            helpers[f"helper_{key}_{case_name}"] = (
                val.detach().cpu().numpy().astype(np.float32, copy=False)
            )
        out_np = out.detach().cpu().numpy().astype(np.float32, copy=False)
        if out_np.shape[1] != len(feature_names):
            raise SystemExit(f"[ERROR] {case_name} 输出 F={out_np.shape[1]} != {len(feature_names)}")
        outputs[f"out_{case_name}"] = out_np
        case_meta.append(
            {
                "case": case_name,
                "n": int(out_np.shape[0]),
                "f": int(out_np.shape[1]),
                "t": int(out_np.shape[2]),
                "atol": payload["atol"],
                "rtol": payload["rtol"],
                "nan_count": int(np.isnan(out_np).sum()),
                "inf_count": int(np.isinf(out_np).sum()),
            }
        )

    args.inputs.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.inputs, **inputs)
    np.savez_compressed(args.baseline, **{**outputs, **helpers})

    inputs_meta = {
        "_comment": "由 scripts/gen_feature_baseline.py 用冻结 AM 生成，请勿手工编辑。",
        "source": "AlphaMaster-main/model_core/features.py",
        "seed": feature_cases.SEED,
        "dtype": "float32",
        "fields": list(feature_cases.FIELDS),
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "cases": case_meta,
        "entries": len(inputs),
    }
    with args.inputs_meta.open("w", encoding="utf-8") as fh:
        json.dump(inputs_meta, fh, ensure_ascii=False, indent=2)
        fh.write("
")

    baseline_meta = {
        "_comment": "AM 原实现 compute_features 输出基准，请勿手工编辑。",
        "source": "AlphaMaster-main/model_core/features.py",
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "dtype": "float32",
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "cases": [c["case"] for c in case_meta],
        "shapes": {c["case"]: [c["n"], c["f"], c["t"]] for c in case_meta},
        "helper_keys": [k for k, _ in feature_cases.HELPER_CASES],
        "entries": len(outputs) + len(helpers),
    }
    with args.baseline_meta.open("w", encoding="utf-8") as fh:
        json.dump(baseline_meta, fh, ensure_ascii=False, indent=2)
        fh.write("
")

    print(f"[OK] AM 根: {am_root}")
    print(f"[OK] torch={torch.__version__} numpy={np.__version__}")
    print(f"[OK] 特征 {len(feature_names)} 个 × 用例 {len(cases)} 个")
    print(f"[OK] 写出: {args.inputs}")
    print(f"[OK] 写出: {args.inputs_meta}")
    print(f"[OK] 写出: {args.baseline}")
    print(f"[OK] 写出: {args.baseline_meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
