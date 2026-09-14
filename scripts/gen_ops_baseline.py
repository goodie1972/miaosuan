# -*- coding: utf-8 -*-
# Copyright (c) 2026 Original Author
# Copyright (c) 2026 Your Company
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""生成 M3 算子对拍的 **torch 基准**（运行在装有真实 torch 的解释器上）。

用 AM 的**原始 torch 实现**（``model_core.ops.OPS_CONFIG``）对确定性用例逐算子求值，
把输入与输出存成 ``tests/fixtures/ops_baseline.npz``，并附 ``ops_baseline_meta.json``。

妙算侧（``tests/parity/test_ops_parity.py``）在**无 torch** 的 venv 中读取该基准，
用 numpy 实现复算并比对 ``max|Δ|``。两侧解耦。

运行：

    C:\Users\Administrator\.workbuddy\binaries\python\envs\default\Scripts\python.exe \
        scripts/gen_ops_baseline.py

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
_OUT_NPZ = _ROOT / "tests" / "fixtures" / "ops_baseline.npz"
_OUT_META = _ROOT / "tests" / "fixtures" / "ops_baseline_meta.json"


def _am_root() -> Path:
    env = os.environ.get("MIAOSUAN_AM_ROOT")
    return Path(env) if env else _DEFAULT_AM_ROOT


def main() -> int:
    parser = argparse.ArgumentParser(description="生成算子对拍 torch 基准")
    parser.add_argument("--am-root", type=Path, default=None)
    parser.add_argument("--npz", type=Path, default=_OUT_NPZ)
    parser.add_argument("--meta", type=Path, default=_OUT_META)
    args = parser.parse_args()

    am_root = args.am_root or _am_root()

    # 共享用例（纯 numpy）
    sys.path.insert(0, str(_ROOT / "tests" / "parity"))
    import ops_cases  # noqa: PLC0415

    # 导入 AM（torch 实现）
    import torch  # noqa: PLC0415

    prev = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(am_root))
    try:
        from model_core.ops import OPS_CONFIG  # noqa: PLC0415
    finally:
        sys.path.remove(str(am_root))
        sys.dont_write_bytecode = prev

    cases = ops_cases.build_cases()
    arrays: dict[str, np.ndarray] = {}
    per_op_report: list[dict] = []

    for case_name, payload in cases.items():
        x = payload["x"]
        y = payload["y"]
        z = payload["z"]
        arrays[f"in_{case_name}_x"] = x
        arrays[f"in_{case_name}_y"] = y
        arrays[f"in_{case_name}_z"] = z

        operands_np = [x, y, z]
        for name, transform, arity in OPS_CONFIG:
            torch_ops = [torch.from_numpy(np.ascontiguousarray(o)) for o in operands_np[:arity]]
            with torch.no_grad():
                out = transform(*torch_ops)
            out_np = out.detach().cpu().numpy().astype(np.float32, copy=False)
            arrays[f"out_{name}_{case_name}"] = out_np
            per_op_report.append(
                {
                    "op": name,
                    "case": case_name,
                    "shape": list(out_np.shape),
                    "nan_count": int(np.isnan(out_np).sum()),
                    "inf_count": int(np.isinf(out_np).sum()),
                }
            )

    args.npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.npz, **arrays)

    meta = {
        "_comment": "由 scripts/gen_ops_baseline.py 用 AM 原始 torch 实现生成，请勿手工编辑。",
        "source": "AlphaMaster-main/model_core/ops.py",
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "dtype": "float32",
        "operator_count": len(OPS_CONFIG),
        "operator_names": [name for name, _t, _a in OPS_CONFIG],
        "cases": list(cases.keys()),
        "nan_case_exempt_ops": sorted(ops_cases.NAN_CASE_EXEMPT_OPS),
        "entries": len(arrays),
    }
    with args.meta.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
        fh.write("
")

    print(f"[OK] AM 根: {am_root}")
    print(f"[OK] torch={torch.__version__} numpy={np.__version__}")
    print(f"[OK] 算子 {len(OPS_CONFIG)} 个 × 用例 {len(cases)} 个 -> {len(arrays)} 个数组")
    print(f"[OK] 写出: {args.npz}")
    print(f"[OK] 写出: {args.meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
