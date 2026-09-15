# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""特征工程（numpy 化移植，M4）—— 65 个特征，纯 numpy，无 torch。

本模块把冻结 ``AlphaMaster`` 的 ``model_core/features.py``（原 torch 实现，
``MT5FeatureEngineer``，65 个特征）**逐特征**移植为纯 numpy 实现，注册进模块级
:data:`FEATURE_REGISTRY`。``compute_features`` 按注册顺序堆叠输出 ``[N, F, T]``
（``F == 65``，维序与 AM **逐元素一致**）。

移植铁律（架构 §1.3、M4 任务书）：

* **名称与顺序一字不改**（65 个特征），否则 ``VOCAB_VERSION`` 会漂移、旧产物被
  ``verify()`` 拒绝；由 ``tests/fixtures/frozen_token_order.json`` 与 ``tests/parity``
  回归锁定（实测保持 ``v9217a2c0d91a``）。
* **数值细节逐一对齐**，不做「顺手优化」：dtype 保持、``clamp`` 边界、``nan``/``inf``
  处理、除零保护 ``_EPS``、``nan_to_num`` 的默认 ``posinf``/``neginf`` 语义、
  ``torch.std`` 的**无偏**（``ddof=1``）默认、``torch.median`` 偶数窗口取**下中位数**
  的语义，均与 AM 一致。
* **禁止** import torch、禁止文件 IO、禁止读取环境变量（CI 的
  ``tests/test_dependency_direction.py`` 强制校验）。active_features 白名单改为
  **参数注入**（``compute_features(..., active_features=...)`` / ``build_feature_registry``），
  **默认注册全部 65 个特征**，从而 import 期零 IO。

torch → numpy 关键等价：

  ``torch.cat([pad, x], 1).unfold(1, w, 1)`` → ``sliding_window_view(concat([pad, x]), w, axis=1)``
  ``torch.clamp(v, lo, hi)``                → ``np.clip(v, lo, hi)``（NaN 均传播）
  ``torch.nan_to_num(x, nan=a, posinf=b, neginf=c)`` → ``_clean`` / ``_nan0``（见下）
  ``torch.median(wnd, dim=-1).values``      → ``_lower_median``（偶数窗口取下中位数）
  ``torch.std(dim=0)``                      → ``ndarray.std(axis=0, ddof=1)``（无偏）
  ``torch.sign(NaN) == 0.0``                → ``_tsign`` = ``(x>0)-(x<0)``
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np

from .registry import FeatureSpec, Registry

__all__ = [
    "FEATURE_REGISTRY",
    "FEATURE_NAMES",
    "FEATURE_COUNT",
    "FEATURE_DEFS",
    "MT5FeatureEngineer",
    "build_feature_registry",
    "compute_features",
    "is_registered",
]

# ── 与 AM 逐项一致的常数（架构 §R1.10；hard constraint：不得改动）──────────
_CLIP_BOUND = 5.0
_EPS = 1e-9
_MA_WINDOW = 20
_NORM_WINDOW = 200


# ── 基础工具 ──────────────────────────────────────────────────────────────


def _f32(x: np.ndarray) -> np.ndarray:
    """等价 ``torch.Tensor.float()``：转 float32（保持 [N, T] 形状）。"""
    return np.asarray(x, dtype=np.float32)


def _clean(x: np.ndarray) -> np.ndarray:
    """等价 ``torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)``。"""
    out: np.ndarray = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def _nan0(x: np.ndarray) -> np.ndarray:
    """等价 ``torch.nan_to_num(x, nan=0.0)``（posinf/neginf 默认取 dtype 极值）。"""
    out: np.ndarray = np.nan_to_num(x, nan=0.0)
    return out


def _win(x: np.ndarray, pad_len: int, width: int) -> np.ndarray:
    """因果左零填充 + 滑窗，返回 ``[N, T, width]``。

    等价 ``torch.cat([zeros(N, pad_len), x], 1).unfold(1, width, 1)``；当
    ``pad_len == width - 1`` 时窗口数恰为 T。
    """
    n = x.shape[0]
    pad = np.zeros((n, pad_len), dtype=x.dtype)
    padded = np.concatenate([pad, x], axis=1)
    return np.lib.stride_tricks.sliding_window_view(padded, width, axis=1)


def _lower_median(a: np.ndarray) -> np.ndarray:
    """沿最后一维取**下中位数**（等价 ``torch.median(dim=-1).values``）。

    偶数长度窗口 torch 返回两中位中**较小**者（sorted 索引 ``n/2-1``）；numpy 的
    ``np.median`` 取均值，语义不同，故此处用 ``partition`` 精确对齐。
    """
    n = a.shape[-1]
    k = (n - 1) // 2
    part = np.partition(a, k, axis=-1)
    out: np.ndarray = part[..., k]
    return out


def _tsign(x: np.ndarray) -> np.ndarray:
    """符号函数，等价 torch.sign 的特殊值语义（``sign(NaN)=0``）。"""
    out: np.ndarray = (x > 0).astype(x.dtype) - (x < 0).astype(x.dtype)
    return out


# ── 滚动统计 helper（逐行对齐 AM）─────────────────────────────────────────


def _rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    out: np.ndarray = _win(x, w - 1, w).mean(axis=-1)
    return out


def _ma(x: np.ndarray, w: int) -> np.ndarray:
    return _rolling_mean(x, w)


def _ma20(x: np.ndarray) -> np.ndarray:
    return _rolling_mean(x, _MA_WINDOW)


def _rolling_sum(x: np.ndarray, w: int) -> np.ndarray:
    out: np.ndarray = _win(x, w - 1, w).sum(axis=-1)
    return out


def _rolling_std(x: np.ndarray, w: int) -> np.ndarray:
    wnd = _win(x, w - 1, w)
    m = wnd.mean(axis=-1, keepdims=True)
    out: np.ndarray = np.sqrt(((wnd - m) ** 2).mean(axis=-1)) + 1e-9
    return out


def _rolling_max(x: np.ndarray, w: int) -> np.ndarray:
    out: np.ndarray = _win(x, w - 1, w).max(axis=-1)
    return out


def _rolling_min(x: np.ndarray, w: int) -> np.ndarray:
    out: np.ndarray = _win(x, w - 1, w).min(axis=-1)
    return out


def _atr(close: np.ndarray, high: np.ndarray, low: np.ndarray, w: int = 14) -> np.ndarray:
    pc = np.concatenate([close[:, :1], close[:, :-1]], axis=1)
    tr = np.stack([high - low, np.abs(high - pc), np.abs(low - pc)], axis=-1).max(axis=-1)
    out: np.ndarray = _rolling_mean(tr, w)
    return out


def _rvol(close: np.ndarray, w: int = 20) -> np.ndarray:
    ret = np.log(close[:, 1:] / (close[:, :-1] + _EPS))
    ret = np.concatenate([np.zeros_like(close[:, :1]), ret], axis=1)
    wnd = _win(ret, w - 1, w)
    m = wnd.mean(axis=-1, keepdims=True)
    out: np.ndarray = np.sqrt(((wnd - m) ** 2).mean(axis=-1)) + 1e-9
    return out


def _ac1(close: np.ndarray, w: int = 20) -> np.ndarray:
    ret = np.log(close[:, 1:] / (close[:, :-1] + _EPS))
    ret = np.concatenate([np.zeros_like(close[:, :1]), ret], axis=1)
    wnd = _win(ret, w, w + 1)
    x, y = wnd[:, :, :-1], wnd[:, :, 1:]
    xm, ym = x.mean(axis=-1, keepdims=True), y.mean(axis=-1, keepdims=True)
    cov = ((x - xm) * (y - ym)).mean(axis=-1)
    sx = np.sqrt(((x - xm) ** 2).mean(axis=-1))
    sy = np.sqrt(((y - ym) ** 2).mean(axis=-1))
    out = cov / (sx * sy + 1e-8)
    return _clean(out)


def _linear_slope(x: np.ndarray, w: int) -> np.ndarray:
    """因果线性回归斜率（按价位归一），出口 ``nan_to_num(nan=0)``。"""
    wnd = _win(x, w - 1, w)
    tidx = np.arange(w, dtype=x.dtype)
    tc = tidx - tidx.mean()
    tvar = (tc**2).sum()
    xm = wnd.mean(axis=-1, keepdims=True)
    slope = ((wnd - xm) * tc).sum(axis=-1) / (tvar + _EPS)
    slope = slope / (xm[..., 0] + _EPS)
    return _nan0(slope)


def _rsi(close: np.ndarray, w: int = 14) -> np.ndarray:
    diff = close - np.concatenate([close[:, :1], close[:, :-1]], axis=1)
    gains = np.maximum(diff, 0.0)
    losses = np.maximum(-diff, 0.0)
    avg_g = _rolling_mean(gains, w)
    avg_l = _rolling_mean(losses, w)
    rs = (avg_g + 1e-9) / (avg_l + 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    out: np.ndarray = (rsi - 50.0) / 50.0
    return out


def _ts_corr(x: np.ndarray, y: np.ndarray, w: int) -> np.ndarray:
    wx = _win(x, w - 1, w)
    wy = _win(y, w - 1, w)
    mx, my = wx.mean(axis=-1, keepdims=True), wy.mean(axis=-1, keepdims=True)
    cov = ((wx - mx) * (wy - my)).mean(axis=-1)
    sx = np.sqrt(((wx - mx) ** 2).mean(axis=-1))
    sy = np.sqrt(((wy - my) ** 2).mean(axis=-1))
    mask = (sx < 1e-6) | (sy < 1e-6)
    corr = cov / (sx * sy + 1e-8)
    corr = np.where(mask, 0.0, corr)
    return _clean(corr)


def _robust_norm(x: np.ndarray, w: int = _NORM_WINDOW) -> np.ndarray:
    """因果滚动 robust 归一化（median/MAD），warm-up 期（t<w-1）输出 0。

    对齐 AM ``_robust_norm``：median/MAD 用**下中位数**（torch 语义），并统一转
    float32 计算后再转回原 dtype（torch 对半精度 median 有精度问题）。
    """
    orig_dtype = x.dtype
    x32 = x.astype(np.float32)
    n, t = x32.shape
    if t < w:
        return np.zeros_like(x)
    wnd = _win(x32, w - 1, w)
    med = _lower_median(wnd)
    mad = _lower_median(np.abs(wnd - med[..., None])) + 1e-6
    out = np.clip((x32 - med) / mad, -_CLIP_BOUND, _CLIP_BOUND)
    out = _clean(out)
    out[:, : w - 1] = 0.0
    return out.astype(orig_dtype)


def _norm(x: np.ndarray) -> np.ndarray:
    """出口 robust_norm 归一化：clip 到 [-5,5]，出入口各 clean 一次。"""
    return _clean(_robust_norm(_clean(x), _NORM_WINDOW))


def _ema_simple(x: np.ndarray, span: int, exact: bool = False) -> np.ndarray:
    """指数加权移动平均（因果），span 期；双路径与 AM **逐分支一致**。

    * ``exact=True``：严格递推 ``out[t]=alpha*x[t]+(1-alpha)*out[t-1]``。
    * 默认路径：``alpha>=1`` 直接返回；``T < 2*w_full`` 走递推；``T >= 2*w_full``
      走向量化首值填充卷积近似。两条路径的切换阈值与 AM 完全相同，保证
      train-serve 数值一致（AM P1-10 修复）。
    """
    alpha = 2.0 / (span + 1.0)
    n, t = x.shape

    if exact:
        out = np.zeros_like(x)
        out[:, 0] = x[:, 0]
        for i in range(1, t):
            out[:, i] = alpha * x[:, i] + (1 - alpha) * out[:, i - 1]
        return out

    if alpha >= 1.0:
        return x.copy()

    # w_full 仅由 span 决定，不依赖 T，保证因果性
    w_full = max(1, math.ceil(-math.log(1e-6) / (-math.log(1.0 - alpha))))

    if t < 2 * w_full:
        out = np.zeros_like(x)
        out[:, 0] = x[:, 0]
        for i in range(1, t):
            out[:, i] = alpha * x[:, i] + (1 - alpha) * out[:, i - 1]
        return out

    decay = 1.0 - alpha
    powers = np.arange(w_full - 1, -1, -1, dtype=x.dtype)
    weights = alpha * (decay**powers)  # 未归一化
    first = np.repeat(x[:, :1], w_full - 1, axis=1)  # [N, w_full-1] 首值填充
    xp = np.concatenate([first, x], axis=1)
    windows = np.lib.stride_tricks.sliding_window_view(xp, w_full, axis=1)  # [N, T, w_full]
    out = (windows * weights).sum(axis=-1)
    return _clean(out)


# ── v3.0 新增特征 helper ──────────────────────────────────────────────────


def _vwap_dev(
    close: np.ndarray, high: np.ndarray, low: np.ndarray, volume: np.ndarray, w: int = 20
) -> np.ndarray:
    typical = (high + low + close) / 3.0
    tpv = typical * volume
    vol_w = _win(volume, w - 1, w)
    tpv_w = _win(tpv, w - 1, w)
    vwap = tpv_w.sum(axis=-1) / (vol_w.sum(axis=-1) + _EPS)
    out: np.ndarray = (close - vwap) / (vwap + _EPS)
    return out


def _boll_pos(close: np.ndarray, w: int = 20) -> tuple[np.ndarray, np.ndarray]:
    ma = _rolling_mean(close, w)
    std = _rolling_std(close, w)
    upper = ma + 2 * std
    lower = ma - 2 * std
    pos = np.clip((close - lower) / (upper - lower + _EPS), 0.0, 1.0)
    width = (upper - lower) / (ma + _EPS)
    return pos, width


def _macd_hist(close: np.ndarray) -> np.ndarray:
    macd = _ema_simple(close, 12) - _ema_simple(close, 26)
    signal = _ema_simple(macd, 9)
    out: np.ndarray = macd - signal
    return out


def _obv_slope(close: np.ndarray, volume: np.ndarray, w: int = 20) -> np.ndarray:
    ret_sign = _tsign(close[:, 1:] - close[:, :-1])
    ret_sign = np.concatenate([np.zeros_like(close[:, :1]), ret_sign], axis=1)
    obv = np.cumsum(ret_sign * volume, axis=1)
    return _linear_slope(obv, w)


def _mfi(
    close: np.ndarray, high: np.ndarray, low: np.ndarray, volume: np.ndarray, w: int = 14
) -> np.ndarray:
    typical = (high + low + close) / 3.0
    mf = typical * volume
    pc = np.concatenate([typical[:, :1], typical[:, :-1]], axis=1)
    pos_mf = np.where(typical > pc, mf, np.zeros_like(mf))
    neg_mf = np.where(typical < pc, mf, np.zeros_like(mf))
    pos_sum = _rolling_mean(pos_mf, w) * w
    neg_sum = _rolling_mean(neg_mf, w) * w
    mfr = pos_sum / (neg_sum + _EPS)
    mfi = 100.0 - (100.0 / (1.0 + mfr))
    out: np.ndarray = (mfi - 50.0) / 50.0
    return out


def _willr(close: np.ndarray, high: np.ndarray, low: np.ndarray, w: int = 14) -> np.ndarray:
    hw = _win(high, w - 1, w).max(axis=-1)
    lw = _win(low, w - 1, w).min(axis=-1)
    willr = (hw - close) / (hw - lw + _EPS)
    out: np.ndarray = np.clip(willr, -1.0, 0.0)
    return out


def _cci(close: np.ndarray, high: np.ndarray, low: np.ndarray, w: int = 14) -> np.ndarray:
    typical = (high + low + close) / 3.0
    ma = _rolling_mean(typical, w)
    tw = _win(typical, w - 1, w)
    mad = np.abs(tw - tw.mean(axis=-1, keepdims=True)).mean(axis=-1)
    cci = (typical - ma) / (0.015 * mad + _EPS)
    out: np.ndarray = np.clip(cci / 200.0, -1.0, 1.0)
    return out


def _roc(close: np.ndarray, w: int = 12) -> np.ndarray:
    raw = close[:, w:] / (close[:, :-w] + _EPS) - 1.0
    pad = np.zeros((close.shape[0], w), dtype=close.dtype)
    return np.concatenate([pad, raw], axis=1)


def _typical_dev(close: np.ndarray, high: np.ndarray, low: np.ndarray, w: int = 20) -> np.ndarray:
    typical = (high + low + close) / 3.0
    ma = _rolling_mean(typical, w)
    out: np.ndarray = (typical - ma) / (ma + _EPS)
    return out


# ── task 5.2 趋势/动量类 helper ───────────────────────────────────────────


def _trend_strength(x: np.ndarray, w: int) -> np.ndarray:
    """SLOPE_w * R²：因果窗口线性回归斜率（按价位归一）乘拟合优度 R²∈[0,1]。"""
    wnd = _win(x, w - 1, w)
    tidx = np.arange(w, dtype=x.dtype)
    tc = tidx - tidx.mean()
    tvar = (tc**2).sum()
    xm = wnd.mean(axis=-1, keepdims=True)
    slope = ((wnd - xm) * tc).sum(axis=-1) / (tvar + _EPS)
    pred = xm + slope[..., None] * tc
    ss_res = ((wnd - pred) ** 2).sum(axis=-1)
    ss_tot = ((wnd - xm) ** 2).sum(axis=-1)
    r2 = np.clip(1.0 - ss_res / (ss_tot + _EPS), 0.0, 1.0)
    slope_norm = slope / (xm[..., 0] + _EPS)
    return _clean(slope_norm * r2)


def _trix(close: np.ndarray, span: int = 15) -> np.ndarray:
    e3 = _ema_simple(_ema_simple(_ema_simple(close, span), span), span)
    prev = np.concatenate([e3[:, :1], e3[:, :-1]], axis=1)
    out: np.ndarray = (e3 - prev) / (np.abs(prev) + _EPS)
    return out


def _ppo(close: np.ndarray) -> np.ndarray:
    e12 = _ema_simple(close, 12)
    e26 = _ema_simple(close, 26)
    out: np.ndarray = (e12 - e26) / (np.abs(e26) + _EPS)
    return out


def _ult_osc(close: np.ndarray, high: np.ndarray, low: np.ndarray) -> np.ndarray:
    pc = np.concatenate([close[:, :1], close[:, :-1]], axis=1)
    true_low = np.minimum(low, pc)
    true_high = np.maximum(high, pc)
    bp = close - true_low
    tr = true_high - true_low

    def _avg(w: int) -> np.ndarray:
        out: np.ndarray = _rolling_sum(bp, w) / (_rolling_sum(tr, w) + _EPS)
        return out

    uo = (4.0 * _avg(7) + 2.0 * _avg(14) + _avg(28)) / 7.0
    out: np.ndarray = np.clip(uo * 2.0 - 1.0, -1.0, 1.0)
    return out


# ── per-feature compute（签名 (raw_dict) -> [N, T]）───────────────────────
# 每个 compute 与原 compute_features 的对应内联片段逐元素等价。共享中间量的特征
# （PV_CORR / REL_*）在各自 compute 内重算其依赖，保持数值一致。


# 趋势类 trend (0-4)
def _c_ret(raw: dict[str, Any]) -> np.ndarray:
    close = _f32(raw["close"])
    n = close.shape[0]
    ret_raw = np.log(close[:, 1:] / (close[:, :-1] + _EPS))
    pad = np.zeros((n, 1), dtype=close.dtype)
    return _norm(np.concatenate([pad, ret_raw], axis=1))


def _c_ret5(raw: dict[str, Any]) -> np.ndarray:
    close = _f32(raw["close"])
    n = close.shape[0]
    ret5 = np.log(close[:, 5:] / (close[:, :-5] + _EPS))
    pad = np.zeros((n, 5), dtype=close.dtype)
    return _norm(np.concatenate([pad, ret5], axis=1))


def _c_ret20(raw: dict[str, Any]) -> np.ndarray:
    close = _f32(raw["close"])
    n = close.shape[0]
    ret20 = np.log(close[:, 20:] / (close[:, :-20] + _EPS))
    pad = np.zeros((n, 20), dtype=close.dtype)
    return _norm(np.concatenate([pad, ret20], axis=1))


def _c_ma_diff(raw: dict[str, Any]) -> np.ndarray:
    close = _f32(raw["close"])
    out: np.ndarray = _ma(close, 10) / (_ma(close, 30) + _EPS) - 1.0
    return _norm(out)


def _c_slope20(raw: dict[str, Any]) -> np.ndarray:
    close = _f32(raw["close"])
    return _norm(_linear_slope(close, 20))


# 波动类 volatility (5-8)
def _c_atr(raw: dict[str, Any]) -> np.ndarray:
    close, high, low = _f32(raw["close"]), _f32(raw["high"]), _f32(raw["low"])
    atr_raw = _atr(close, high, low)
    return _norm(np.log1p(_clean(np.clip(atr_raw, 0.0, None))))


def _c_rvol(raw: dict[str, Any]) -> np.ndarray:
    close = _f32(raw["close"])
    rvol_raw = _rvol(close)
    return _norm(np.log1p(_clean(np.clip(rvol_raw, 0.0, None))))


def _c_hl_range(raw: dict[str, Any]) -> np.ndarray:
    close, high, low = _f32(raw["close"]), _f32(raw["high"]), _f32(raw["low"])
    out: np.ndarray = (high - low) / (close + _EPS)
    return _norm(out)


def _c_vol_regime(raw: dict[str, Any]) -> np.ndarray:
    close, high, low = _f32(raw["close"]), _f32(raw["high"]), _f32(raw["low"])
    atr_raw = _atr(close, high, low)
    ma_atr = _ma(atr_raw, 20)
    out: np.ndarray = atr_raw / (ma_atr + _EPS) - 1.0
    return _norm(out)


# 反转类 reversal (9-13)
def _c_dev(raw: dict[str, Any]) -> np.ndarray:
    close = _f32(raw["close"])
    ma20c = _ma20(close)
    out: np.ndarray = (close - ma20c) / (ma20c + _EPS)
    return _norm(out)


def _c_dev60(raw: dict[str, Any]) -> np.ndarray:
    close = _f32(raw["close"])
    ma60 = _ma(close, 60)
    out: np.ndarray = (close - ma60) / (ma60 + _EPS)
    return _norm(out)


def _c_rsi14(raw: dict[str, Any]) -> np.ndarray:
    close = _f32(raw["close"])
    return _clean(np.clip(_rsi(close, 14), -1.0, 1.0))


def _c_pressure(raw: dict[str, Any]) -> np.ndarray:
    close, open_ = _f32(raw["close"]), _f32(raw["open"])
    high, low = _f32(raw["high"]), _f32(raw["low"])
    # P2-13：high==low（无波动）时输出 0（中性），避免无信息极值污染因子
    hl_range = high - low
    no_range = np.abs(hl_range) < _EPS
    safe_range = np.where(no_range, np.ones_like(hl_range), hl_range)
    ratio = (close - open_) / safe_range
    ratio = np.where(no_range, np.zeros_like(ratio), ratio)
    return _clean(np.clip(ratio, -1.0, 1.0))


def _c_ac1(raw: dict[str, Any]) -> np.ndarray:
    close = _f32(raw["close"])
    return _clean(np.clip(_ac1(close), -1.0, 1.0))


# 成交量类 volume (14-16)
def _c_vol_ratio(raw: dict[str, Any]) -> np.ndarray:
    volume = _f32(raw["volume"])
    ma20v = _ma20(volume)
    out: np.ndarray = volume / (ma20v + _EPS)
    return _norm(out)


def _c_vol_z(raw: dict[str, Any]) -> np.ndarray:
    volume = _f32(raw["volume"])
    ma20v = _ma20(volume)
    std20v = _rolling_std(volume, 20)
    out: np.ndarray = (volume - ma20v) / (std20v + _EPS)
    return _clean(np.clip(out, -5.0, 5.0))


def _c_pv_corr(raw: dict[str, Any]) -> np.ndarray:
    # 依赖归一化后的 RET（feature 0）与 VOL_RATIO（feature 14），逐元素重算
    ret = _c_ret(raw)
    vol_ratio = _c_vol_ratio(raw)
    log_vol_ratio = np.log1p(_clean(np.clip(vol_ratio, -0.99, None)))
    return _clean(np.clip(_ts_corr(ret, log_vol_ratio, 10), -1.0, 1.0))


# 跨截面相对强弱 cross_sectional (17-19)
# N=1（单品种）下截面去均值得到 0 常数，退化为时间序列归一化以保留信息量。
def _c_rel_ret5(raw: dict[str, Any]) -> np.ndarray:
    ret5 = _c_ret5(raw)
    if ret5.shape[0] == 1:
        return _clean(ret5)
    out: np.ndarray = ret5 - ret5.mean(axis=0, keepdims=True)
    return _norm(out)


def _c_rel_ret20(raw: dict[str, Any]) -> np.ndarray:
    ret20 = _c_ret20(raw)
    if ret20.shape[0] == 1:
        return _clean(ret20)
    out: np.ndarray = ret20 - ret20.mean(axis=0, keepdims=True)
    return _norm(out)


def _c_rel_vol(raw: dict[str, Any]) -> np.ndarray:
    rvol = _c_rvol(raw)
    if rvol.shape[0] == 1:
        return _clean(rvol)
    out: np.ndarray = rvol - rvol.mean(axis=0, keepdims=True)
    return _norm(out)


# v3.0 新增特征 (20-25)
def _c_vwap_dev(raw: dict[str, Any]) -> np.ndarray:
    close, high, low = _f32(raw["close"]), _f32(raw["high"]), _f32(raw["low"])
    volume = _f32(raw["volume"])
    return _norm(_vwap_dev(close, high, low, volume))


def _c_boll_pos(raw: dict[str, Any]) -> np.ndarray:
    close = _f32(raw["close"])
    boll_pos, _ = _boll_pos(close)
    return _clean(boll_pos)


def _c_boll_width(raw: dict[str, Any]) -> np.ndarray:
    close = _f32(raw["close"])
    _, boll_width = _boll_pos(close)
    return _norm(boll_width)


def _c_macd_hist(raw: dict[str, Any]) -> np.ndarray:
    close = _f32(raw["close"])
    return _norm(_macd_hist(close))


def _c_obv_slope(raw: dict[str, Any]) -> np.ndarray:
    close, volume = _f32(raw["close"]), _f32(raw["volume"])
    return _norm(_obv_slope(close, volume))


def _c_mfi14(raw: dict[str, Any]) -> np.ndarray:
    close, high, low = _f32(raw["close"]), _f32(raw["high"]), _f32(raw["low"])
    volume = _f32(raw["volume"])
    return _clean(np.clip(_mfi(close, high, low, volume), -1.0, 1.0))


# v3.0 Alpha 101 + 互补特征 (26-29)
def _c_willr14(raw: dict[str, Any]) -> np.ndarray:
    close, high, low = _f32(raw["close"]), _f32(raw["high"]), _f32(raw["low"])
    return _clean(_willr(close, high, low))


def _c_cci14(raw: dict[str, Any]) -> np.ndarray:
    close, high, low = _f32(raw["close"]), _f32(raw["high"]), _f32(raw["low"])
    return _clean(_cci(close, high, low))


def _c_roc12(raw: dict[str, Any]) -> np.ndarray:
    close = _f32(raw["close"])
    return _norm(_roc(close, 12))


def _c_typical_dev(raw: dict[str, Any]) -> np.ndarray:
    close, high, low = _f32(raw["close"]), _f32(raw["high"]), _f32(raw["low"])
    return _norm(_typical_dev(close, high, low))


# ── task 5.2 趋势类 trend (30-32) ─────────────────────────────────────────
def _c_ema_ratio_12_26(raw: dict[str, Any]) -> np.ndarray:
    close = _f32(raw["close"])
    e12 = _ema_simple(close, 12)
    e26 = _ema_simple(close, 26)
    out: np.ndarray = e12 / (e26 + _EPS) - 1.0
    return _norm(out)


def _c_trend_strength_50(raw: dict[str, Any]) -> np.ndarray:
    close = _f32(raw["close"])
    return _norm(_trend_strength(close, 50))


def _c_price_pos_50(raw: dict[str, Any]) -> np.ndarray:
    close, high, low = _f32(raw["close"]), _f32(raw["high"]), _f32(raw["low"])
    w = 50
    max50 = _win(high, w - 1, w).max(axis=-1)
    min50 = _win(low, w - 1, w).min(axis=-1)
    pos = (close - min50) / (max50 - min50 + _EPS)
    return _clean(np.clip(pos, 0.0, 1.0))


# ── task 5.3 波动类（含 OHLC 估计量）volatility (37-40) ───────────────────
def _c_gk_vol(raw: dict[str, Any]) -> np.ndarray:
    open_ = _f32(raw["open"])
    high, low = _f32(raw["high"]), _f32(raw["low"])
    close = _f32(raw["close"])
    ln2 = math.log(2.0)
    hl_term = 0.5 * (np.log((high + _EPS) / (low + _EPS))) ** 2
    co_term = (2.0 * ln2 - 1.0) * (np.log((close + _EPS) / (open_ + _EPS))) ** 2
    gk_bar = np.clip(hl_term - co_term, 0.0, None)
    gk_mean = _rolling_mean(gk_bar, 20)
    raw_vol = np.sqrt(gk_mean)
    return _norm(np.log1p(_clean(raw_vol)))


def _c_parkinson_vol(raw: dict[str, Any]) -> np.ndarray:
    high, low = _f32(raw["high"]), _f32(raw["low"])
    ln2 = math.log(2.0)
    pk_bar = (1.0 / (4.0 * ln2)) * (np.log((high + _EPS) / (low + _EPS))) ** 2
    pk_bar = np.clip(pk_bar, 0.0, None)
    pk_mean = _rolling_mean(pk_bar, 20)
    raw_vol = np.sqrt(pk_mean)
    return _norm(np.log1p(_clean(raw_vol)))


def _c_yang_zhang_vol(raw: dict[str, Any]) -> np.ndarray:
    open_ = _f32(raw["open"])
    high, low = _f32(raw["high"]), _f32(raw["low"])
    close = _f32(raw["close"])
    pc = np.concatenate([close[:, :1], close[:, :-1]], axis=1)  # 前收盘（因果）
    overnight_bar = (np.log((open_ + _EPS) / (pc + _EPS))) ** 2
    open_bar = (np.log((open_ + _EPS) / (close + _EPS))) ** 2
    rs_bar = np.log((high + _EPS) / (close + _EPS)) * np.log(
        (high + _EPS) / (open_ + _EPS)
    ) + np.log((low + _EPS) / (close + _EPS)) * np.log((low + _EPS) / (open_ + _EPS))
    rs_bar = np.clip(rs_bar, 0.0, None)
    yz_bar = (overnight_bar + open_bar + rs_bar) / 3.0
    yz_mean = _rolling_mean(yz_bar, 20)
    raw_vol = np.sqrt(np.clip(yz_mean, 0.0, None))
    return _norm(np.log1p(_clean(raw_vol)))


def _c_rs_vol(raw: dict[str, Any]) -> np.ndarray:
    open_ = _f32(raw["open"])
    high, low = _f32(raw["high"]), _f32(raw["low"])
    close = _f32(raw["close"])
    rs_bar = np.log((high + _EPS) / (close + _EPS)) * np.log(
        (high + _EPS) / (open_ + _EPS)
    ) + np.log((low + _EPS) / (close + _EPS)) * np.log((low + _EPS) / (open_ + _EPS))
    rs_mean = _rolling_mean(rs_bar, 20)
    raw_vol = np.sqrt(np.clip(rs_mean, 0.0, None))
    return _norm(np.log1p(_clean(raw_vol)))


# ── task 5.2 动量类 momentum (33-36) ──────────────────────────────────────
def _c_trix_15(raw: dict[str, Any]) -> np.ndarray:
    close = _f32(raw["close"])
    return _norm(_trix(close, 15))


def _c_ppo(raw: dict[str, Any]) -> np.ndarray:
    close = _f32(raw["close"])
    return _norm(_ppo(close))


def _c_ult_osc(raw: dict[str, Any]) -> np.ndarray:
    close, high, low = _f32(raw["close"]), _f32(raw["high"]), _f32(raw["low"])
    return _clean(_ult_osc(close, high, low))


def _c_ret_accel(raw: dict[str, Any]) -> np.ndarray:
    close = _f32(raw["close"])
    n = close.shape[0]
    ret5 = np.log(close[:, 5:] / (close[:, :-5] + _EPS))
    pad = np.zeros((n, 5), dtype=close.dtype)
    ret5 = np.concatenate([pad, ret5], axis=1)
    prev = np.concatenate([pad, ret5[:, :-5]], axis=1)  # ret5[t-5]，前 5 位为 0
    out: np.ndarray = ret5 - prev
    return _norm(out)


# ── task 5.4 量能/流动性类 volume (41-44) ─────────────────────────────────
def _c_amihud_illiq(raw: dict[str, Any]) -> np.ndarray:
    close, volume = _f32(raw["close"]), _f32(raw["volume"])
    n = close.shape[0]
    log_ret = np.log(close[:, 1:] / (close[:, :-1] + _EPS))
    log_ret = np.concatenate([np.zeros((n, 1), dtype=close.dtype), log_ret], axis=1)
    illiq_bar = np.abs(log_ret) / (volume + _EPS)
    illiq_mean = _rolling_mean(illiq_bar, 20)
    return _norm(np.log1p(_clean(np.clip(illiq_mean, 0.0, None))))


def _c_kyle_lambda(raw: dict[str, Any]) -> np.ndarray:
    close, volume = _f32(raw["close"]), _f32(raw["volume"])
    n = close.shape[0]
    w = 20
    log_ret = np.log(close[:, 1:] / (close[:, :-1] + _EPS))
    log_ret = np.concatenate([np.zeros((n, 1), dtype=close.dtype), log_ret], axis=1)
    abs_ret = np.abs(log_ret)
    signed_vol = _tsign(log_ret) * volume
    wa = _win(abs_ret, w - 1, w)
    ws = _win(signed_vol, w - 1, w)
    ma = wa.mean(axis=-1, keepdims=True)
    ms = ws.mean(axis=-1, keepdims=True)
    cov = ((wa - ma) * (ws - ms)).mean(axis=-1)
    var_s = ((ws - ms) ** 2).mean(axis=-1)
    lam = cov / (var_s + _EPS)
    return _norm(_clean(lam))


def _c_cmf_20(raw: dict[str, Any]) -> np.ndarray:
    close, high, low = _f32(raw["close"]), _f32(raw["high"]), _f32(raw["low"])
    volume = _f32(raw["volume"])
    mf_mul = ((close - low) - (high - close)) / (high - low + _EPS)
    mfv = mf_mul * volume
    cmf = _rolling_sum(mfv, 20) / (_rolling_sum(volume, 20) + _EPS)
    return _clean(np.clip(cmf, -1.0, 1.0))


def _c_ad_line_slope(raw: dict[str, Any]) -> np.ndarray:
    close, high, low = _f32(raw["close"]), _f32(raw["high"]), _f32(raw["low"])
    volume = _f32(raw["volume"])
    mf_mul = ((close - low) - (high - close)) / (high - low + _EPS)
    mfv = mf_mul * volume
    ad_line = np.cumsum(mfv, axis=1)
    slope = _linear_slope(ad_line, 20)
    return _norm(_clean(slope))


# ── task 5.5 反转/振荡类特征 (45-50) ──────────────────────────────────────
def _c_stoch_k_14(raw: dict[str, Any]) -> np.ndarray:
    close, high, low = _f32(raw["close"]), _f32(raw["high"]), _f32(raw["low"])
    w = 14
    max_h = _win(high, w - 1, w).max(axis=-1)
    min_l = _win(low, w - 1, w).min(axis=-1)
    pct_k = (close - min_l) / (max_h - min_l + _EPS)
    out = np.clip(pct_k * 2.0 - 1.0, -1.0, 1.0)
    return _clean(out)


def _c_stoch_d_3(raw: dict[str, Any]) -> np.ndarray:
    close, high, low = _f32(raw["close"]), _f32(raw["high"]), _f32(raw["low"])
    w = 14
    max_h = _win(high, w - 1, w).max(axis=-1)
    min_l = _win(low, w - 1, w).min(axis=-1)
    pct_k = (close - min_l) / (max_h - min_l + _EPS)
    pct_d = _rolling_mean(pct_k, 3)
    out = np.clip(pct_d * 2.0 - 1.0, -1.0, 1.0)
    return _clean(out)


def _c_aroon_osc_25(raw: dict[str, Any]) -> np.ndarray:
    high, low = _f32(raw["high"]), _f32(raw["low"])
    w = 25
    wh = _win(high, w - 1, w)
    wl = _win(low, w - 1, w)
    idx_h = wh.argmax(axis=-1).astype(high.dtype)
    idx_l = wl.argmin(axis=-1).astype(low.dtype)
    periods_h = (w - 1) - idx_h
    periods_l = (w - 1) - idx_l
    aroon_up = (w - periods_h) / w
    aroon_down = (w - periods_l) / w
    osc = aroon_up - aroon_down
    return _clean(np.clip(osc, -1.0, 1.0))


def _c_dmi_adx_14(raw: dict[str, Any]) -> np.ndarray:
    close, high, low = _f32(raw["close"]), _f32(raw["high"]), _f32(raw["low"])
    w = 14
    prev_h = np.concatenate([high[:, :1], high[:, :-1]], axis=1)
    prev_l = np.concatenate([low[:, :1], low[:, :-1]], axis=1)
    prev_c = np.concatenate([close[:, :1], close[:, :-1]], axis=1)
    dm_pos = np.clip(high - prev_h, 0.0, None)
    dm_neg = np.clip(prev_l - low, 0.0, None)
    tr = np.stack([high - low, np.abs(high - prev_c), np.abs(low - prev_c)], axis=-1).max(axis=-1)
    tr_mean = _rolling_mean(tr, w)
    di_pos = _rolling_mean(dm_pos, w) / (tr_mean + _EPS)
    di_neg = _rolling_mean(dm_neg, w) / (tr_mean + _EPS)
    dx = np.abs(di_pos - di_neg) / (di_pos + di_neg + _EPS)
    adx = _rolling_mean(dx, w)
    return _clean(np.clip(adx, 0.0, 1.0))


def _c_dmi_diff_14(raw: dict[str, Any]) -> np.ndarray:
    close, high, low = _f32(raw["close"]), _f32(raw["high"]), _f32(raw["low"])
    w = 14
    prev_h = np.concatenate([high[:, :1], high[:, :-1]], axis=1)
    prev_l = np.concatenate([low[:, :1], low[:, :-1]], axis=1)
    prev_c = np.concatenate([close[:, :1], close[:, :-1]], axis=1)
    dm_pos = np.clip(high - prev_h, 0.0, None)
    dm_neg = np.clip(prev_l - low, 0.0, None)
    tr = np.stack([high - low, np.abs(high - prev_c), np.abs(low - prev_c)], axis=-1).max(axis=-1)
    tr_mean = _rolling_mean(tr, w)
    di_pos = _rolling_mean(dm_pos, w) / (tr_mean + _EPS)
    di_neg = _rolling_mean(dm_neg, w) / (tr_mean + _EPS)
    diff = di_pos - di_neg
    return _clean(np.clip(diff, -1.0, 1.0))


def _c_trix_signal(raw: dict[str, Any]) -> np.ndarray:
    close = _f32(raw["close"])
    trix = _trix(close, 15)
    signal = _rolling_mean(trix, 9)
    return _norm(_clean(trix - signal))


# ── task 5.6 通道/突破类 channel (51-56) ──────────────────────────────────
def _c_donchian_pos_20(raw: dict[str, Any]) -> np.ndarray:
    close, high, low = _f32(raw["close"]), _f32(raw["high"]), _f32(raw["low"])
    max20 = _rolling_max(high, 20)
    min20 = _rolling_min(low, 20)
    pos = (close - min20) / (max20 - min20 + _EPS)
    return _clean(np.clip(pos, 0.0, 1.0))


def _c_keltner_pos_20(raw: dict[str, Any]) -> np.ndarray:
    close, high, low = _f32(raw["close"]), _f32(raw["high"]), _f32(raw["low"])
    mid = _ema_simple(close, 20)
    atr = _atr(close, high, low, w=14)
    rng = _ema_simple(atr, 20)
    upper = mid + 2.0 * rng
    lower = mid - 2.0 * rng
    pos = (close - lower) / (upper - lower + _EPS)
    return _clean(np.clip(pos, 0.0, 1.0))


def _c_ichimoku_kijun_dev(raw: dict[str, Any]) -> np.ndarray:
    close, high, low = _f32(raw["close"]), _f32(raw["high"]), _f32(raw["low"])
    kijun = (_rolling_max(high, 26) + _rolling_min(low, 26)) / 2.0
    dev = (close - kijun) / (kijun + _EPS)
    return _norm(_clean(dev))


def _c_ichimoku_tenkan_dev(raw: dict[str, Any]) -> np.ndarray:
    close, high, low = _f32(raw["close"]), _f32(raw["high"]), _f32(raw["low"])
    tenkan = (_rolling_max(high, 9) + _rolling_min(low, 9)) / 2.0
    dev = (close - tenkan) / (tenkan + _EPS)
    return _norm(_clean(dev))


def _c_supertrend_dir(raw: dict[str, Any]) -> np.ndarray:
    close, high, low = _f32(raw["close"]), _f32(raw["high"]), _f32(raw["low"])
    n, t = close.shape
    atr = _atr(close, high, low, w=14)
    mid = (high + low) / 2.0
    upper_band = mid + 1.5 * atr
    lower_band = mid - 1.5 * atr
    # P2-15：初始化 0（中性），直到首次突破才确定方向；严格因果（仅用 t 与 t-1 带值）
    direction = np.zeros((n, t), dtype=close.dtype)
    prev_upper = upper_band[:, 0]
    prev_lower = lower_band[:, 0]
    for i in range(1, t):
        flip_up = close[:, i] > prev_upper
        flip_down = close[:, i] < prev_lower
        new_dir = direction[:, i - 1].copy()
        new_dir[flip_up] = 1.0
        new_dir[flip_down] = -1.0
        direction[:, i] = new_dir
        prev_upper = upper_band[:, i]
        prev_lower = lower_band[:, i]
    return _clean(direction)


def _c_sar_dist(raw: dict[str, Any]) -> np.ndarray:
    close = _f32(raw["close"])
    ema20 = _ema_simple(close, 20)
    ema50 = _ema_simple(close, 50)
    sar_approx = ema20 - ema50
    return _norm(_clean(sar_approx))


# ── task 5.7 统计类 statistical (57-62) ───────────────────────────────────
def _c_roll_skew_20(raw: dict[str, Any]) -> np.ndarray:
    close = _f32(raw["close"])
    n = close.shape[0]
    ret = np.log(close[:, 1:] / (close[:, :-1] + _EPS))
    ret = np.concatenate([np.zeros((n, 1), dtype=close.dtype), ret], axis=1)
    w = 20
    wnd = _win(ret, w - 1, w)
    mean = wnd.mean(axis=-1, keepdims=True)
    diff = wnd - mean
    std = np.sqrt((diff**2).mean(axis=-1))
    skew = (diff**3).mean(axis=-1) / (std**3 + _EPS)
    return _norm(_clean(skew))


def _c_roll_kurt_20(raw: dict[str, Any]) -> np.ndarray:
    close = _f32(raw["close"])
    n = close.shape[0]
    ret = np.log(close[:, 1:] / (close[:, :-1] + _EPS))
    ret = np.concatenate([np.zeros((n, 1), dtype=close.dtype), ret], axis=1)
    w = 20
    wnd = _win(ret, w - 1, w)
    mean = wnd.mean(axis=-1, keepdims=True)
    diff = wnd - mean
    std = np.sqrt((diff**2).mean(axis=-1))
    kurt = (diff**4).mean(axis=-1) / (std**4 + _EPS) - 3.0
    return _norm(_clean(kurt))


def _c_hurst_50(raw: dict[str, Any]) -> np.ndarray:
    close = _f32(raw["close"])
    n = close.shape[0]
    ret = np.log(close[:, 1:] / (close[:, :-1] + _EPS))
    ret = np.concatenate([np.zeros((n, 1), dtype=close.dtype), ret], axis=1)
    w = 50
    wnd = _win(ret, w - 1, w)
    mean = wnd.mean(axis=-1, keepdims=True)
    centered = wnd - mean
    cumdev = np.cumsum(centered, axis=-1)
    r = cumdev.max(axis=-1) - cumdev.min(axis=-1)
    s = np.sqrt((centered**2).mean(axis=-1))
    # P2-14：常数窗口（R/S 过小）输出 0.5（中性），避免 log≈-13.8 -> -1 的强信号污染
    no_signal = (r < _EPS) | (s < _EPS)
    safe_s = np.where(no_signal, np.ones_like(s), s)
    safe_r = np.where(no_signal, np.ones_like(r), r)
    hurst = np.log(safe_r / (safe_s + _EPS) + _EPS) / math.log(w)
    hurst = np.clip(hurst, 0.0, 1.0)
    hurst = np.where(no_signal, np.full_like(hurst, 0.5), hurst)
    return _clean(hurst * 2.0 - 1.0)


def _c_fractal_dim_30(raw: dict[str, Any]) -> np.ndarray:
    close = _f32(raw["close"])
    n = close.shape[0]
    w = 30
    wnd_c = _win(close, w - 1, w)
    rng = wnd_c.max(axis=-1) - wnd_c.min(axis=-1)
    diff = np.abs(close[:, 1:] - close[:, :-1])
    diff = np.concatenate([np.zeros((n, 1), dtype=close.dtype), diff], axis=1)
    wnd_d = _win(diff, w - 1, w)
    mad = wnd_d.mean(axis=-1)
    frac = rng / (mad * math.sqrt(w) + _EPS)
    frac = np.clip(frac, 0.0, 3.0)
    return _clean(frac / 3.0 * 2.0 - 1.0)


def _c_ac2(raw: dict[str, Any]) -> np.ndarray:
    close = _f32(raw["close"])
    n = close.shape[0]
    ret = np.log(close[:, 1:] / (close[:, :-1] + _EPS))
    ret = np.concatenate([np.zeros((n, 1), dtype=close.dtype), ret], axis=1)
    w = 20
    lag = 2
    wnd = _win(ret, w + lag - 1, w + lag)
    x = wnd[:, :, :-lag]
    y = wnd[:, :, lag:]
    xm = x.mean(axis=-1, keepdims=True)
    ym = y.mean(axis=-1, keepdims=True)
    cov = ((x - xm) * (y - ym)).mean(axis=-1)
    sx = np.sqrt(((x - xm) ** 2).mean(axis=-1))
    sy = np.sqrt(((y - ym) ** 2).mean(axis=-1))
    corr = cov / (sx * sy + 1e-8)
    return _clean(np.clip(corr, -1.0, 1.0))


def _c_ret_entropy_20(raw: dict[str, Any]) -> np.ndarray:
    close = _f32(raw["close"])
    n = close.shape[0]
    ret = np.log(close[:, 1:] / (close[:, :-1] + _EPS))
    ret = np.concatenate([np.zeros((n, 1), dtype=close.dtype), ret], axis=1)
    w = 20
    wnd = _win(ret, w - 1, w)
    p_pos = (wnd > 0).astype(ret.dtype).mean(axis=-1)
    p_neg = (wnd < 0).astype(ret.dtype).mean(axis=-1)
    p_zero = (wnd == 0).astype(ret.dtype).mean(axis=-1)

    def _h(p: np.ndarray) -> np.ndarray:
        out: np.ndarray = -p * np.log(p + _EPS)
        return out

    entropy = _h(p_pos) + _h(p_neg) + _h(p_zero)
    return _clean(np.clip(entropy / math.log(3), 0.0, 1.0))


# ── task 5.8 跨截面相对强弱类特征（补充）(63-64) ──────────────────────────
def _c_cs_rank_ret5(raw: dict[str, Any]) -> np.ndarray:
    close = _f32(raw["close"])
    n, t = close.shape
    ret5 = np.log(close[:, 5:] / (close[:, :-5] + _EPS))
    ret5 = np.concatenate([np.zeros((n, 5), dtype=close.dtype), ret5], axis=1)
    if n == 1:
        return _clean(_norm(ret5))
    order = np.argsort(ret5, axis=0, kind="stable")
    ranks = np.empty_like(ret5)
    idx_vals = np.broadcast_to(np.arange(n, dtype=ret5.dtype).reshape(n, 1), (n, t))
    np.put_along_axis(ranks, order, idx_vals, axis=0)
    cs_rank = ranks / (n - 1)
    return _clean(np.clip(cs_rank, 0.0, 1.0))


def _c_cs_zscore_ret20(raw: dict[str, Any]) -> np.ndarray:
    close = _f32(raw["close"])
    n = close.shape[0]
    ret20 = np.log(close[:, 20:] / (close[:, :-20] + _EPS))
    ret20 = np.concatenate([np.zeros((n, 20), dtype=close.dtype), ret20], axis=1)
    if n == 1:
        return _clean(_norm(ret20))
    cs_mean = ret20.mean(axis=0, keepdims=True)
    # torch.std 默认无偏（ddof=1）
    cs_std = ret20.std(axis=0, keepdims=True, ddof=1) + _EPS
    zscore = (ret20 - cs_mean) / cs_std
    return _norm(_clean(zscore))


# ── 注册（顺序即 token/特征维顺序，0..64）─────────────────────────────────

# (name, category, compute) —— 逐项对齐 AM _FEATURE_DEFS，名称/顺序不可变更。
_FEATURE_DEFS: list[tuple[str, str, Any]] = [
    # 趋势类 trend (0-4)
    ("RET", "trend", _c_ret),
    ("RET5", "trend", _c_ret5),
    ("RET20", "trend", _c_ret20),
    ("MA_DIFF", "trend", _c_ma_diff),
    ("SLOPE20", "trend", _c_slope20),
    # 波动类 volatility (5-8)
    ("ATR", "volatility", _c_atr),
    ("RVOL", "volatility", _c_rvol),
    ("HL_RANGE", "volatility", _c_hl_range),
    ("VOL_REGIME", "volatility", _c_vol_regime),
    # 反转类 reversal (9-13)
    ("DEV", "reversal", _c_dev),
    ("DEV60", "reversal", _c_dev60),
    ("RSI14", "reversal", _c_rsi14),
    ("PRESSURE", "reversal", _c_pressure),
    ("AC1", "reversal", _c_ac1),
    # 成交量类 volume (14-16)
    ("VOL_RATIO", "volume", _c_vol_ratio),
    ("VOL_Z", "volume", _c_vol_z),
    ("PV_CORR", "volume", _c_pv_corr),
    # 跨截面相对强弱 cross_sectional (17-19)
    ("REL_RET5", "cross_sectional", _c_rel_ret5),
    ("REL_RET20", "cross_sectional", _c_rel_ret20),
    ("REL_VOL", "cross_sectional", _c_rel_vol),
    # v3.0 新增特征 (20-25)
    ("VWAP_DEV", "volume", _c_vwap_dev),
    ("BOLL_POS", "channel", _c_boll_pos),
    ("BOLL_WIDTH", "volatility", _c_boll_width),
    ("MACD_HIST", "momentum", _c_macd_hist),
    ("OBV_SLOPE", "volume", _c_obv_slope),
    ("MFI14", "volume", _c_mfi14),
    # v3.0 Alpha 101 + 互补特征 (26-29)
    ("WILLR_14", "reversal", _c_willr14),
    ("CCI_14", "reversal", _c_cci14),
    ("ROC_12", "momentum", _c_roc12),
    ("TYPICAL_DEV", "reversal", _c_typical_dev),
    # task 5.2 趋势类 trend (30-32)
    ("EMA_RATIO_12_26", "trend", _c_ema_ratio_12_26),
    ("TREND_STRENGTH_50", "trend", _c_trend_strength_50),
    ("PRICE_POS_50", "trend", _c_price_pos_50),
    # task 5.2 动量类 momentum (33-36)
    ("TRIX_15", "momentum", _c_trix_15),
    ("PPO", "momentum", _c_ppo),
    ("ULT_OSC", "momentum", _c_ult_osc),
    ("RET_ACCEL", "momentum", _c_ret_accel),
    # task 5.3 波动类（含 OHLC 估计量）volatility (37-40)
    ("GK_VOL", "volatility", _c_gk_vol),
    ("PARKINSON_VOL", "volatility", _c_parkinson_vol),
    ("YANG_ZHANG_VOL", "volatility", _c_yang_zhang_vol),
    ("RS_VOL", "volatility", _c_rs_vol),
    # task 5.4 量能/流动性类 volume (41-44)
    ("AMIHUD_ILLIQ", "volume", _c_amihud_illiq),
    ("KYLE_LAMBDA", "volume", _c_kyle_lambda),
    ("CMF_20", "volume", _c_cmf_20),
    ("AD_LINE_SLOPE", "volume", _c_ad_line_slope),
    # task 5.5 反转/振荡类 reversal/trend/momentum (45-50)
    ("STOCH_K_14", "reversal", _c_stoch_k_14),
    ("STOCH_D_3", "reversal", _c_stoch_d_3),
    ("AROON_OSC_25", "reversal", _c_aroon_osc_25),
    ("DMI_ADX_14", "trend", _c_dmi_adx_14),
    ("DMI_DIFF_14", "trend", _c_dmi_diff_14),
    ("TRIX_SIGNAL", "momentum", _c_trix_signal),
    # task 5.6 通道/突破类 channel (51-56)
    ("DONCHIAN_POS_20", "channel", _c_donchian_pos_20),
    ("KELTNER_POS_20", "channel", _c_keltner_pos_20),
    ("ICHIMOKU_KIJUN_DEV", "channel", _c_ichimoku_kijun_dev),
    ("ICHIMOKU_TENKAN_DEV", "channel", _c_ichimoku_tenkan_dev),
    ("SUPERTREND_DIR", "channel", _c_supertrend_dir),
    ("SAR_DIST", "channel", _c_sar_dist),
    # task 5.7 统计类 statistical (57-62)
    ("ROLL_SKEW_20", "statistical", _c_roll_skew_20),
    ("ROLL_KURT_20", "statistical", _c_roll_kurt_20),
    ("HURST_50", "statistical", _c_hurst_50),
    ("FRACTAL_DIM_30", "statistical", _c_fractal_dim_30),
    ("AC2", "statistical", _c_ac2),
    ("RET_ENTROPY_20", "statistical", _c_ret_entropy_20),
    # task 5.8 跨截面相对强弱补充 cross_sectional (63-64)
    ("CS_RANK_RET5", "cross_sectional", _c_cs_rank_ret5),
    ("CS_ZSCORE_RET20", "cross_sectional", _c_cs_zscore_ret20),
]

#: 只读的 (name, category) 视图，便于外部校验名称/顺序。
FEATURE_DEFS: tuple[tuple[str, str], ...] = tuple(
    (name, category) for name, category, _compute in _FEATURE_DEFS
)


def build_feature_registry(active_features: Iterable[str] | None = None) -> Registry:
    """构造特征注册表。

    :param active_features: 若为 ``None``（默认），注册**全部 65 个**特征；否则仅注册
        白名单内的特征（按 ``_FEATURE_DEFS`` 原始顺序过滤，保持维序稳定）。这是
        AM ``active_features.json`` 剪枝机制的**参数注入**替代：core 不做 import 期 IO。
    :returns: 新的 :class:`Registry`（Feature 条目按顺序注册）。
    """
    registry = Registry()
    allow = None if active_features is None else {str(n) for n in active_features}
    for name, category, compute in _FEATURE_DEFS:
        if allow is not None and name not in allow:
            continue
        registry.register_feature(FeatureSpec(name=name, category=category, compute=compute))
    return registry


#: 默认注册表（全部 65 个特征，import 期零 IO）。
FEATURE_REGISTRY: Registry = build_feature_registry()

#: 由注册表导出有序特征名视图（保持下游 import 兼容）。
FEATURE_NAMES: tuple[str, ...] = FEATURE_REGISTRY.feature_names

#: 特征总数（= 65，与 AM 一致）。
FEATURE_COUNT: int = len(FEATURE_REGISTRY.feature_names)


def is_registered(name: str) -> bool:
    """该 token 名称是否已注册（跨 Feature/Operator 全局唯一）。"""
    return name in FEATURE_REGISTRY


def compute_features(
    raw_dict: dict[str, Any], active_features: Iterable[str] | None = None
) -> np.ndarray:
    """按注册顺序计算全部（或白名单）特征，堆叠为 ``[N, F, T]``。

    逐特征 compute 的数值与顺序与 AM 逐元素一致；出口统一 ``nan_to_num``（nan/inf→0）。

    :param raw_dict: 含 ``close/high/low/open/volume``（各 ``[N, T]``）的原始行情。
    :param active_features: 可选特征白名单（参数注入）；``None`` 时使用全部 65 个。
    """
    registry = (
        FEATURE_REGISTRY if active_features is None else build_feature_registry(active_features)
    )
    feats = [spec.compute(raw_dict) for spec in registry.feature_specs]
    features = np.stack(feats, axis=1)
    return _clean(features)


class MT5FeatureEngineer:
    """特征引擎命名空间（AM API 兼容）。

    提供默认特征维度 :attr:`INPUT_DIM` 与 :meth:`compute_features` 静态入口；
    全部数值计算由模块级 :func:`compute_features` 与 ``_c_*`` 实现。
    """

    INPUT_DIM: int = FEATURE_COUNT

    @staticmethod
    def compute_features(
        raw_dict: dict[str, Any], active_features: Iterable[str] | None = None
    ) -> np.ndarray:
        """等价于模块级 :func:`compute_features`（AM 静态方法入口兼容）。"""
        return compute_features(raw_dict, active_features=active_features)
