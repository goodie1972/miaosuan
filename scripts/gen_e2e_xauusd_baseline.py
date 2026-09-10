"""生成**端到端对拍**基准：XAUUSD_H1 真实数据 × best_XAUUSD 真实公式（torch Oracle）。

基准 = 冻结 AM 的**原生管线**：

    ParquetDataManager.load()  →  raw_dict
    MT5FeatureEngineer.compute_features(raw)  →  feat [1, 65, T]
    StackVM().execute(formula, feat)          →  factor [1, T]

产出 ``tests/fixtures/e2e_xauusd_baseline.npz``（原始 OHLCV 面板 + 特征张量 + 因子序列）与
``tests/fixtures/e2e_xauusd_meta.json``（形状、公式、统计量：mean/std/min/max/在市占比）。

妙算侧（``tests/parity/test_e2e_xauusd.py``）用同一 parquet 自行重建 raw_dict（校验数据路径），
再跑妙算 numpy 全链路（``compute_features`` → ``StackVM.execute``），逐点比对 ``max|Δ|`` 并
核对统计量。

运行（Oracle python，装 torch）：

    C:/Users/Administrator/.workbuddy/binaries/python/envs/default/Scripts/python.exe \\
        scripts/gen_e2e_xauusd_baseline.py

对 AM 仓库**零写入**。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_AM_ROOT = Path(r"D:\backup\BaoBao\PythonProgram\AlphaMaster-main")
_DEFAULT_PARQUET = Path(r"D:\K线数据\XAUUSD_H1.parquet")
_FIX = _ROOT / "tests" / "fixtures"
_OUT_NPZ = _FIX / "e2e_xauusd_baseline.npz"
_OUT_JSON = _FIX / "e2e_xauusd_meta.json"
_FIELDS = ("open", "high", "low", "close", "volume")


def _am_root() -> Path:
    env = os.environ.get("MIAOSUAN_AM_ROOT")
    return Path(env) if env else _DEFAULT_AM_ROOT


def _stats(factor: np.ndarray) -> dict[str, float]:
    pos = np.tanh(factor)
    return {
        "mean": float(np.mean(factor)),
        "std": float(np.std(factor)),
        "min": float(np.min(factor)),
        "max": float(np.max(factor)),
        "in_market_ratio": float(np.mean(np.abs(pos) >= 0.05)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="生成端到端 XAUUSD torch 基准")
    parser.add_argument("--am-root", type=Path, default=None)
    parser.add_argument("--parquet", type=Path, default=_DEFAULT_PARQUET)
    parser.add_argument("--npz", type=Path, default=_OUT_NPZ)
    parser.add_argument("--json", type=Path, default=_OUT_JSON)
    args = parser.parse_args()

    am_root = args.am_root or _am_root()
    parquet = args.parquet

    prev = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(am_root))
    try:
        from data_pipeline.parquet_manager import ParquetDataManager  # noqa: PLC0415
        from model_core.vm import StackVM  # noqa: PLC0415
        from model_core.features import MT5FeatureEngineer  # noqa: PLC0415
    finally:
        sys.path.remove(str(am_root))
        sys.dont_write_bytecode = prev

    import torch  # noqa: PLC0415

    strategy = json.loads(
        (am_root / "strategies" / "best_XAUUSD.json").read_text(encoding="utf-8")
    )
    formula = list(strategy["formula"])

    mgr = ParquetDataManager(str(parquet))
    mgr.load()
    raw = mgr.raw_dict  # {field: torch [1, T]} + time

    with torch.no_grad():
        feat = MT5FeatureEngineer.compute_features(raw)  # [1, 65, T]
        factor = StackVM().execute(formula, feat)  # [1, T] or None

    if factor is None:
        raise SystemExit("[ERROR] AM StackVM.execute 返回 None（公式不可求值？）")

    out: dict[str, np.ndarray] = {}
    for field in _FIELDS:
        out[f"raw_{field}"] = raw[field].detach().cpu().numpy().astype(np.float32, copy=False)
    out["raw_time"] = raw["time"].detach().cpu().numpy().astype(np.int64, copy=False)
    out["feat"] = feat.detach().cpu().numpy().astype(np.float32, copy=False)
    factor_np = factor.detach().cpu().numpy().astype(np.float32, copy=False)
    out["factor"] = factor_np

    t_bars = int(factor_np.shape[1])
    meta: dict[str, Any] = {
        "_comment": "由 scripts/gen_e2e_xauusd_baseline.py 用冻结 AM 原生管线生成，请勿手工编辑。",
        "source": "ParquetDataManager → MT5FeatureEngineer.compute_features → StackVM.execute",
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "parquet": str(parquet),
        "symbol": strategy.get("symbol"),
        "timeframe": strategy.get("timeframe"),
        "vocab_version": strategy.get("vocab_version"),
        "formula": formula,
        "formula_decoded": strategy.get("formula_decoded"),
        "bars": t_bars,
        "feature_shape": list(feat.shape),
        "factor_shape": list(factor_np.shape),
        "am_stats": _stats(factor_np),
    }

    # 数据路径事实（供数据适配坑清单核对）
    import pandas as pd  # noqa: PLC0415

    df = pd.read_parquet(parquet)
    meta["data_facts"] = {
        "rows_raw": int(len(df)),
        "rows_used": t_bars,
        "columns": list(df.columns),
        "time_min": int(df["time"].min()),
        "time_max": int(df["time"].max()),
        "has_tick_volume": bool("tick_volume" in df.columns),
    }

    args.npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.npz, **out)
    with args.json.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    print(f"[OK] AM 根: {am_root}")
    print(f"[OK] parquet: {parquet} bars={t_bars}")
    print(f"[OK] formula={formula} (vocab={meta['vocab_version']})")
    print(f"[OK] AM stats={meta['am_stats']}")
    print(f"[OK] 写出: {args.npz}")
    print(f"[OK] 写出: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
