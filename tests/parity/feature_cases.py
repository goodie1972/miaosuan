# -*- coding: utf-8 -*-
# Copyright (c) 2026 Original Author
# Copyright (c) 2026 Your Company
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""M4 特征对拍的**确定性输入用例**（纯 numpy，无 torch / 无妙算依赖）。

同一份用例被两侧共享：

* ``scripts/gen_feature_baseline.py``（torch 环境）→ 生成 AM 原实现的输入/输出基准；
* ``tests/parity/test_feature_parity.py``（妙算 venv）→ 用妙算 numpy 实现复算并比对。

由于两侧 numpy 版本一致（2.5.3）且 seed 固定，``build_cases()`` 在两侧**逐位相同**；
生成器仍会把输入一并写入基准文件，测试侧再做一次输入一致性校验，防止 RNG 漂移。

用例覆盖（含 team-lead 要求的 warm-up / 长序列尾段 / 常量退化 / 单品种退化）：

  ==========  ==============================================  ==============
  case        含义                                             容差 (atol, rtol)
  ==========  ==============================================  ==============
  base        随机游走 O(100)，N=8，T=320（含 warm-up 期）      (3e-3, 1e-2)
  trend       时变漂移强趋势，N=4，T=400（动量类有离散度）      (5e-3, 5e-2)
  long        长序列 N=5，T=800（覆盖 EMA 两路径 + robust_norm）(3e-3, 1e-2)
  single      N=1（跨截面算子退化路径）                        (3e-3, 1e-2)
  short       T=120 < 200（robust_norm 全 warm-up 路径）        (3e-3, 1e-2)
  const       OHLC/量全常数（零跨度退化窗口）                    (1e-3, 1e-3)
  ==========  ==============================================  ==============

容差说明（为何不是 1e-5）
--------------------------

特征输出统一经过 ``_norm``（median/MAD 稳健归一化）。当某特征的**原始**信号在
窗口内离散度很小（如近似单调趋势下的动量类），其 MAD 接近下界，1e-7 量级的
float32 归约顺序噪声会被放大 ``1/MAD`` 倍（可达 1e2~1e3）。这是 float32 归约
顺序差异（torch 与 numpy）经非线性归一化放大的**固有数值现象**，非移植缺陷：
裸 EMA 实测与 torch 相对误差 ≤ 3e-7（见 ``tests/parity/test_feature_helpers.py``）。
故输出层容差按「归一化放大」设定，并在 ``test_feature_helpers.py`` 以 1e-5 的**紧
容差**单独对拍共享底层 helper，两层共同保证移植正确性。
"""

from __future__ import annotations

from typing import Any

import numpy as np

# 全局随机种子（与 AppConfig 默认 seed 一致）
SEED = 20260910

# dtype：与 AM 生产张量一致（float32）
DTYPE = np.float32

#: raw_dict 字段（顺序稳定）
FIELDS: tuple[str, ...] = ("close", "high", "low", "open", "volume")


def _case(raw: dict[str, np.ndarray], atol: float, rtol: float) -> dict[str, Any]:
    payload: dict[str, Any] = {k: raw[k].astype(DTYPE, copy=False) for k in FIELDS}
    payload["atol"] = atol
    payload["rtol"] = rtol
    return payload


def _ohlcv(
    rng: np.random.Generator,
    n: int,
    t: int,
    base: float = 100.0,
    vol: float = 1e5,
    drift: float = 0.0002,
    sigma: float = 0.01,
) -> dict[str, np.ndarray]:
    """构造一致的正价 OHLCV 面板：``high >= max(o,c)``、``low <= min(o,c)``。"""
    steps = rng.normal(drift, sigma, size=(n, t))
    close = base * np.exp(np.cumsum(steps, axis=1))
    open_ = np.empty_like(close)
    open_[:, 0] = close[:, 0]
    open_[:, 1:] = close[:, :-1] * (1.0 + rng.normal(0.0, 0.002, size=(n, t - 1)))
    hi = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, 0.003, size=(n, t))))
    lo = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.003, size=(n, t))))
    high = np.maximum(hi, np.maximum(open_, close))
    low = np.minimum(lo, np.minimum(open_, close))
    volume = np.abs(rng.normal(vol, vol * 0.3, size=(n, t))) + 1.0
    return {"close": close, "high": high, "low": low, "open": open_, "volume": volume}


def _trend(
    rng: np.random.Generator, n: int, t: int, base: float = 100.0, vol: float = 1e5
) -> dict[str, np.ndarray]:
    """时变漂移强趋势序列：动量类特征有真实离散度（MAD 远离下界），数值良态。"""
    tt = np.arange(t, dtype=np.float64)
    drift = 0.0006 + 0.0025 * np.sin(2.0 * np.pi * tt / 60.0)
    steps = drift[None, :] + rng.normal(0.0, 0.004, size=(n, t))
    close = base * np.exp(np.cumsum(steps, axis=1))
    open_ = np.empty_like(close)
    open_[:, 0] = close[:, 0]
    open_[:, 1:] = close[:, :-1] * (1.0 + rng.normal(0.0, 0.002, size=(n, t - 1)))
    hi = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, 0.003, size=(n, t))))
    lo = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.003, size=(n, t))))
    high = np.maximum(hi, np.maximum(open_, close))
    low = np.minimum(lo, np.minimum(open_, close))
    volume = np.abs(rng.normal(vol, vol * 0.25, size=(n, t))) + 1.0
    return {"close": close, "high": high, "low": low, "open": open_, "volume": volume}


def _constant(n: int, t: int, price: float = 100.0, vol: float = 1e5) -> dict[str, np.ndarray]:
    """OHLC 与成交量全常数：制造零跨度 / 退化窗口（high==low）。"""
    arr = np.full((n, t), price, dtype=np.float64)
    volume = np.full((n, t), vol, dtype=np.float64)
    return {
        "close": arr.copy(),
        "high": arr.copy(),
        "low": arr.copy(),
        "open": arr.copy(),
        "volume": volume,
    }


def build_cases() -> dict[str, dict[str, Any]]:
    """构造全部确定性用例，返回 ``{case_name: {字段..., "atol","rtol"}}``。"""
    cases: dict[str, dict[str, Any]] = {}

    cases["base"] = _case(_ohlcv(np.random.default_rng(SEED), 8, 320), atol=3e-3, rtol=1e-2)
    cases["trend"] = _case(_trend(np.random.default_rng(SEED + 1), 4, 400), atol=5e-3, rtol=5e-2)
    cases["long"] = _case(_ohlcv(np.random.default_rng(SEED + 2), 5, 800), atol=3e-3, rtol=1e-2)
    cases["single"] = _case(_ohlcv(np.random.default_rng(SEED + 3), 1, 320), atol=3e-3, rtol=1e-2)
    cases["short"] = _case(_ohlcv(np.random.default_rng(SEED + 4), 3, 120), atol=3e-3, rtol=1e-2)
    cases["const"] = _case(_constant(4, 320), atol=1e-3, rtol=1e-3)

    return cases


#: 共享底层 helper 的对拍输入规格：``(helper_key, args)``，args 取自 case 的 OHLCV。
#: 这些 helper 是全部特征的计算基石，单独以紧容差（1e-5）对拍，隔离「归一化放大」。
HELPER_CASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ema_span12", ("close", 12)),
    ("ema_span15", ("close", 15)),
    ("ema_span20", ("close", 20)),
    ("ema_span26", ("close", 26)),
    ("ema_span50", ("close", 50)),
    ("ma10", ("close", 10)),
    ("rolling_std20", ("close", 20)),
    ("atr14", ("close", "high", "low", 14)),
    ("rvol", ("close",)),
    ("ac1", ("close",)),
    ("linear_slope20", ("close", 20)),
    ("trend_strength50", ("close", 50)),
    ("ts_corr10", ("close", "close", 10)),
    ("robust_norm_close", ("close",)),
    ("trix15", ("close", 15)),
)


def case_names() -> list[str]:
    return list(build_cases().keys())
