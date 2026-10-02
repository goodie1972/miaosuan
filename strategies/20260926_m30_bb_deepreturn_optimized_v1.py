# -*- coding: utf-8 -*-
"""m30_bb_deepreturn_optimized —— 由妙算（MiaoSuan）自动生成，**请勿手工编辑**。

公式（RPN，4 个 token）
-----------------------------------
BOLL_POS → MFI14 → ADD → NEG

语义
----
* 因子：``StackVM`` 求值 + 滚动 500 期 z-score（clip 到 ±3，前 499
  根输出 0）——与 ``miaosuan.core.vm.StackVM`` **逐位同源**（内核源码由 core 原样导出）。
* 仓位：``exposure = tanh(factor)``；``|exposure| < 0.05`` 视为中性、**不出信号**。
* bar1 语义（不重绘）：``generate_signal`` **只用已收盘 K 线**（``self.candles[:-1]``），
  最后一根未收盘 K 线不参与任何计算，因此不存在未来函数。
* 风控（SL/TP）：``get_dynamic_sl_tp`` 按平台实际调用契约实现
  （``main.py:1901/1953`` 传 **2 个位置参数**，返回**绝对价格**）。硬止损
  ``入场价 ∓ 3.0×ATR(14)``，ATR 用的是**真实值**（由已收盘
  K 线算出），不是平台兜底的硬编码 15。
  ``TP_ATR_MULT <= 0`` 表示**无固定止盈**（趋势跟踪：持仓全靠因子反向出场），
  与回测口径一致——回测层没有 SL/TP，设固定止盈会造成实盘与回测行为背离。
  止损 3×ATR 比平台兜底的 2×ATR 更宽，减少噪音扫损。
证据（导出时刻快照，只读）
-------------------------
* 验证集得分：None（0 折 Walk-Forward）
* 去膨胀夏普 DSR：None（试验次数 1）
* 成本敏感度：n/a
* hold-out 夏普：未消费 hold-out
* 门禁结论：RESEARCH_ONLY

溯源
----
* IR spec id：1e01feae75453529
* 生成时刻：2026-09-26T06:29:43.094052+00:00
* git sha：2277081
* 词表版本：v9217a2c0d91a（token 仅在同一词表下有意义）
* 数据指纹：
* 随机种子：0
* 市场/品种：FOREX_XAUUSD
"""
# ═══════════════════════════════════════════════════════════════════════════
# ⚠  RESEARCH_ONLY —— 本策略**未通过实盘门禁**，仅供研究 / 模拟，禁止上实盘。
#
# 门禁结论：RESEARCH_ONLY
# 触发原因：
#   - LLM-assisted conversion from Shenji strategy 20260711_m30_bb_deepreturn_optimized_v1.py
#   - Entry factor only; exit logic delegated to composition layer
#   - Signal overlap verified: 100% match at original entry points (131 long + 105 short)
#   - Trend confirmation + ATR volatility bonus delegated to composition layer
#
# 需要重新过门禁并重新导出后，本文件才会去掉本警示块。
# ═══════════════════════════════════════════════════════════════════════════
from __future__ import annotations

import math
from typing import Any

import numpy as np
from strategies.base import BaseStrategy
from core.bridge import OrderType

# ── 词表版本（硬约束 H-1 冻结锁；lint AF007 会校验与内核派生版本一致）────────
VOCAB_VERSION = "v9217a2c0d91a"

# ── 妙算 Spec 快照（机器可读，供反向提取使用；请勿手工编辑）────
_MIAOSUAN_SPEC_JSON = r"""{
  "spec_version": "1.0",
  "name": "m30_bb_deepreturn_optimized",
  "payload": {
    "kind": "factor",
    "tokens": [
      21,
      25,
      65,
      69
    ],
    "vocab_version": "v9217a2c0d91a"
  },
  "semantics": {
    "position_fn": "tanh",
    "neutral_band": 0.05,
    "long_short": true,
    "warmup_bars": 499,
    "timeframe": "M30",
    "roll_window": 500
  },
  "evidence": {
    "n_trials": 1,
    "wf_folds": 0,
    "val_score": null,
    "holdout_sharpe": null,
    "cost_sensitivity": {},
    "deflated_sharpe": null,
    "gate_verdict": "RESEARCH_ONLY",
    "gate_reasons": [
      "LLM-assisted conversion from Shenji strategy 20260711_m30_bb_deepreturn_optimized_v1.py",
      "Entry factor only; exit logic delegated to composition layer",
      "Signal overlap verified: 100% match at original entry points (131 long + 105 short)",
      "Trend confirmation + ATR volatility bonus delegated to composition layer"
    ]
  },
  "provenance": {
    "git_sha": "2277081",
    "vocab_version": "v9217a2c0d91a",
    "data_fingerprint": "",
    "seed": 0,
    "created_at": "2026-09-26T06:29:43.094052+00:00",
    "miaosuan_version": "0.1.0",
    "market": "FOREX_XAUUSD",
    "budget": "manual",
    "config_snapshot": {
      "source_strategy": "20260711_m30_bb_deepreturn_optimized_v1.py",
      "source_magic": 661102,
      "source_version": "v3_optimized",
      "conversion_method": "LLM-assisted static analysis + empirical overlap test",
      "entry_candidate": "A (pure reversal, 100% overlap)",
      "exit_strategy": "delegated to composition layer (scheme 2)",
      "overlap_test": "131/131 long match, 105/105 short match on XAUUSD_H1 4013 bars"
    }
  },
  "notes": "BB + MFI pure reversal factor. Converted from Shenji strategy 20260711_m30_bb_deepreturn_optimized_v1.py (magic=661102, v3_optimized). RPN: -(BOLL_POS + MFI14). Signal overlap test on XAUUSD H1 data: 100% match at original strategy entry points (131 long entries, 105 short entries). Trend confirmation (EMA_RATIO), directional confirmation (DMI_DIFF), and ATR volatility bonus from the original strategy are delegated to the composition/trading engine layer, as they introduced directional noise when combined additively in the RPN."
}"""

# ── 策略身份（妙算 契约：四项必须齐全）────────────────────────────────
# magic 必须是**裸 int**：平台侧是 magic: int（core/bridge.py:40），
# 写成字符串会导致 p.magic == magic 永不相等 → 接管不到旧仓、产生孤儿单。
STRATEGY_MAGIC = 661102
STRATEGY_NAME = "m30_bb_deepreturn_optimized"
STRATEGY_VERSION = "1"
STRATEGY_CHANGELOG = (
    "v1 由妙算自动生成：BOLL_POS → MFI14 → ADD → NEG",
    "v1 IR spec_id=1e01feae75453529 vocab=v9217a2c0d91a",
    "v1 证据：val_score=None wf_folds=0 n_trials=1 dsr=None gate=RESEARCH_ONLY",
    "v1 溯源：git=2277081 data=n/a seed=0",
)

#: 是否通过实盘门禁（False 时本文件仅供研究）。
DEPLOYABLE = False

#: 供 tune 消费的参数空间（与下方类属性一一对应）。
PARAM_SPACE = (
    {
        "name": "NEUTRAL_BAND",
        "default": 0.05,
        "kind": "float",
        "low": 0.0,
        "high": 0.5,
        "step": 0.005,
        "choices": [],
        "description": "中性带：|tanh(factor)| 小于该值视为不交易",
    },
)

# ═══════════════════════════════════════════════════════════════════════════
# 因子内核 —— 以下代码由 miaosuan.core.{features,ops} **原样导出**并统一加前缀
# 重命名（ft_ / op_），数值与挖掘时逐位一致。**不要手工改动**，否则会破坏
# 与回测的一致性；改动应回到妙算侧重新挖掘后重新导出。
# ═══════════════════════════════════════════════════════════════════════════

ft_EPS = 1e-9

def ft_f32(x: np.ndarray) -> np.ndarray:
    """等价 ``numpy.ndarray.float()``：转 float32（保持 [N, T] 形状）。"""
    return np.asarray(x, dtype=np.float32)

def ft_clean(x: np.ndarray) -> np.ndarray:
    """等价 ``numpy.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)``。"""
    out: np.ndarray = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return out

def ft_win(x: np.ndarray, pad_len: int, width: int) -> np.ndarray:
    """因果左零填充 + 滑窗，返回 ``[N, T, width]``。

    等价 ``numpy.cat([zeros(N, pad_len), x], 1).unfold(1, width, 1)``；当
    ``pad_len == width - 1`` 时窗口数恰为 T。
    """
    n = x.shape[0]
    pad = np.zeros((n, pad_len), dtype=x.dtype)
    padded = np.concatenate([pad, x], axis=1)
    return np.lib.stride_tricks.sliding_window_view(padded, width, axis=1)

def ft_rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    out: np.ndarray = ft_win(x, w - 1, w).mean(axis=-1)
    return out

def ft_rolling_std(x: np.ndarray, w: int) -> np.ndarray:
    wnd = ft_win(x, w - 1, w)
    m = wnd.mean(axis=-1, keepdims=True)
    out: np.ndarray = np.sqrt(((wnd - m) ** 2).mean(axis=-1)) + 1e-9
    return out

def ft_boll_pos(close: np.ndarray, w: int = 20) -> tuple[np.ndarray, np.ndarray]:
    ma = ft_rolling_mean(close, w)
    std = ft_rolling_std(close, w)
    upper = ma + 2 * std
    lower = ma - 2 * std
    pos = np.clip((close - lower) / (upper - lower + ft_EPS), 0.0, 1.0)
    width = (upper - lower) / (ma + ft_EPS)
    return pos, width

def ft_mfi(
    close: np.ndarray, high: np.ndarray, low: np.ndarray, volume: np.ndarray, w: int = 14
) -> np.ndarray:
    typical = (high + low + close) / 3.0
    mf = typical * volume
    pc = np.concatenate([typical[:, :1], typical[:, :-1]], axis=1)
    pos_mf = np.where(typical > pc, mf, np.zeros_like(mf))
    neg_mf = np.where(typical < pc, mf, np.zeros_like(mf))
    pos_sum = ft_rolling_mean(pos_mf, w) * w
    neg_sum = ft_rolling_mean(neg_mf, w) * w
    mfr = pos_sum / (neg_sum + ft_EPS)
    mfi = 100.0 - (100.0 / (1.0 + mfr))
    out: np.ndarray = (mfi - 50.0) / 50.0
    return out

def ft_c_boll_pos(raw: dict[str, Any]) -> np.ndarray:
    close = ft_f32(raw["close"])
    boll_pos, _ = ft_boll_pos(close)
    return ft_clean(boll_pos)

def ft_c_mfi14(raw: dict[str, Any]) -> np.ndarray:
    close, high, low = ft_f32(raw["close"]), ft_f32(raw["high"]), ft_f32(raw["low"])
    volume = ft_f32(raw["volume"])
    return ft_clean(np.clip(ft_mfi(close, high, low, volume), -1.0, 1.0))

lambda_add_480 = lambda x, y: x + y

lambda_neg_484 = lambda x: -x

# ── 公式与算子/特征绑定 ────────────────────────────────────────────────────
_FEAT_OFFSET = 65
_TOKENS = (21, 25, 65, 69)
_FEATURE_FUNCS = {
    21: ft_c_boll_pos,  # BOLL_POS
    25: ft_c_mfi14,  # MFI14
}
def _op_wrap_65(*operands: np.ndarray) -> np.ndarray:
    """算子 ADD（arity=2）的入口包装。"""
    return lambda_add_480(*operands)

def _op_wrap_69(*operands: np.ndarray) -> np.ndarray:
    """算子 NEG（arity=1）的入口包装。"""
    return lambda_neg_484(*operands)


_OP_FUNCS = {
    65: (_op_wrap_65, 2),  # ADD
    69: (_op_wrap_69, 1),  # NEG
}


# ── 栈式求值 + 滚动归一化（对齐 miaosuan.core.vm.StackVM）──────────────────
_ROLL_WINDOW = 500


def _normalize_output(x: np.ndarray, roll_window: int = _ROLL_WINDOW) -> np.ndarray:
    """因果滚动标准化（逐位对齐 ``StackVM._normalize_output``）。"""
    n, t = x.shape
    global_std = float(x.std(ddof=1)) if t > 1 else 0.0
    if global_std < 1e-6:
        return x
    if n > 1:
        cs_mean = x.mean(axis=0, keepdims=True)
        cs_std = np.clip(x.std(axis=0, keepdims=True, ddof=1), 1e-8, None)
        return np.clip((x - cs_mean) / cs_std, -3.0, 3.0)
    _roll = int(roll_window)
    if t < _roll:
        cnt = np.arange(1, t + 1, dtype=x.dtype).reshape(1, t)
        cumsum = np.cumsum(x, axis=1)
        ts_mean = cumsum / cnt
        cumsum_sq = np.cumsum(x * x, axis=1)
        ts_var = (cumsum_sq / cnt) - ts_mean * ts_mean
        ts_std = np.sqrt(np.clip(ts_var, 1e-8, None))
        return np.asarray(np.clip((x - ts_mean) / ts_std, -3.0, 3.0))
    pad = np.zeros((n, _roll - 1), dtype=x.dtype)
    padded = np.concatenate([pad, x], axis=1)
    windows = np.lib.stride_tricks.sliding_window_view(padded, _roll, axis=1)
    ts_mean = windows.mean(axis=2)
    ts_std = np.clip(windows.std(axis=2, ddof=1), 1e-8, None)
    ts_z = (x - ts_mean) / ts_std
    ts_z[:, np.arange(t) < (_roll - 1)] = 0.0
    return np.asarray(np.clip(ts_z, -3.0, 3.0))


def _evaluate_rpn(features: dict[int, np.ndarray], tokens: tuple[int, ...]) -> np.ndarray:
    """前缀（RPN）栈式求值，语义对齐 ``StackVM.execute``。"""
    stack: list[np.ndarray] = []
    for token in tokens:
        if token < _FEAT_OFFSET:
            stack.append(features[token])
        else:
            fn, arity = _OP_FUNCS[token]
            args: list[np.ndarray] = [stack.pop() for _ in range(arity)]
            args.reverse()
            res = fn(*args)
            if np.isnan(res).any() or np.isinf(res).any():
                res = np.nan_to_num(res, nan=0.0, posinf=1.0, neginf=-1.0)
            stack.append(res)
    return stack[0]


def compute_factor(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    gate_deadband: "float | None" = None,
) -> np.ndarray:
    """计算因子序列。

    Args:
        open_: 开盘价，形状 ``[T]``（1D）或 ``[1, T]``。
        high: 最高价。
        low: 最低价。
        close: 收盘价。
        volume: 成交量。

    Returns:
        归一化后的因子，形状 ``[T]``（1D）。前 ``_ROLL_WINDOW - 1`` 根为 0（预热）。

        ``gate_deadband`` 为 ``None`` 时沿用模板渲染的模块级默认值 ``_GATE_DEADBAND``；
        传入则覆盖它，使模式 B（tune 改写类属性 ``GATE_DEADBAND`` 后）经
        :meth:`generate_signal` 注入、真正影响死区计算（否则是死参数）。
    """
    raw = {
        "open": np.atleast_2d(np.asarray(open_, dtype=np.float32)),
        "high": np.atleast_2d(np.asarray(high, dtype=np.float32)),
        "low": np.atleast_2d(np.asarray(low, dtype=np.float32)),
        "close": np.atleast_2d(np.asarray(close, dtype=np.float32)),
        "volume": np.atleast_2d(np.asarray(volume, dtype=np.float32)),
    }
    features = {tid: fn(raw) for tid, fn in _FEATURE_FUNCS.items()}
    value = _evaluate_rpn(features, _TOKENS)
    return _normalize_output(value, _ROLL_WINDOW)[0]


# ── 平台数据适配 ───────────────────────────────────────────────────────────
_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "open": ("open", "o", "Open", "OPEN"),
    "high": ("high", "h", "High", "HIGH"),
    "low": ("low", "l", "Low", "LOW"),
    "close": ("close", "c", "Close", "CLOSE"),
    "volume": ("volume", "v", "tick_volume", "Volume", "VOLUME"),
}


def _candles_to_arrays(candles: Any) -> tuple[Any, ...] | None:
    """把平台 candle 序列转成 ``(open, high, low, close, volume)`` 五个 1D 数组。

    兼容三种常见形态：``pandas.DataFrame`` / ``dict`` 列表 / 对象列表。
    长度不足或字段缺失时返回 ``None``（调用方应放弃本次信号）。

    Args:
        candles: 平台传入的 K 线序列（**只应包含已收盘 K 线**）。

    Returns:
        五元组；失败返回 ``None``。
    """
    if candles is None or len(candles) == 0:
        return None
    try:
        if hasattr(candles, "columns"):
            columns = {str(c).lower(): c for c in candles.columns}
            picked: list[Any] = []
            for key in ("open", "high", "low", "close", "volume"):
                column = None
                for alias in _FIELD_ALIASES[key]:
                    if alias.lower() in columns:
                        column = columns[alias.lower()]
                        break
                if column is None:
                    return None
                picked.append(np.asarray(candles[column]).reshape(-1))
            return tuple(picked)
        first = candles[0]
        if isinstance(first, dict):
            getter = lambda i, k: candles[i][k]  # noqa: E731
        else:
            getter = lambda i, k: getattr(candles[i], k)  # noqa: E731
        out: list[Any] = []
        for key in ("open", "high", "low", "close", "volume"):
            values: list[float] = []
            for i in range(len(candles)):
                chosen = None
                for alias in _FIELD_ALIASES[key]:
                    try:
                        chosen = getter(i, alias)
                        break
                    except (KeyError, AttributeError, TypeError):
                        continue
                values.append(0.0 if chosen is None else float(chosen))
            out.append(np.asarray(values, dtype=np.float64))
        return tuple(out)
    except (KeyError, AttributeError, TypeError, IndexError, ValueError):
        return None


# ── ATR / 方向归一（风控契约）──────────────────────────────────────────────
#: ATR 周期（Wilder，与现网 ``strategies/20260909_h1_alphagate_v1._get_atr`` 一致）。
_ATR_PERIOD = 14

#: 买入方向别名集合（``OrderType`` 枚举 + 字符串两种写法都要认）。
_BUY_DIRECTIONS = frozenset({"BUY", "BUY_LIMIT", "BUY_STOP", "LONG", "1"})


def _is_buy_direction(direction: Any) -> bool:
    """判断 ``direction`` 是否为买入方向（**枚举 / 字符串 / 数值三种都兼容**）。

    为什么必须归一化：平台 ``OrderType`` 是**普通 Enum**（``core/bridge.py:17``），
    ``OrderType.BUY == "BUY"`` 是 ``False``。而引擎 ``main.py:1901`` 传进来的
    正是 ``OrderType.BUY``，现网新一批策略却写成 ``if direction == "BUY"`` ——
    于是 BUY 信号会落进 SELL 分支，**止损挂到入场价上方**（实盘一开仓即被扫）。

    Args:
        direction: ``OrderType`` 枚举 / ``"BUY"`` 之类字符串 / ``1`` 之类数值。

    Returns:
        ``True`` 表示买入方向。
    """
    name = getattr(direction, "name", None)
    if isinstance(name, str):
        return name.strip().upper() in _BUY_DIRECTIONS
    value = getattr(direction, "value", direction)
    if isinstance(value, str):
        return value.strip().upper() in _BUY_DIRECTIONS
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)):
        return float(value) > 0.0
    return False


def _atr_from_arrays(
    high: Any,
    low: Any,
    close: Any,
    period: int = _ATR_PERIOD,
) -> float:
    """由**已收盘** K 线算 ATR（Wilder 口径：最近 ``period`` 根真实波幅的均值）。

    Args:
        high: 最高价序列（``[T]``，时间升序，末位为最新）。
        low: 最低价序列。
        close: 收盘价序列。
        period: ATR 周期。

    Returns:
        ATR（价格单位）；数据不足 / 非有限 / 非正时返回 ``0.0``，
        由调用方决定兜底（**绝不用一个假的正数蒙混过去**）。
    """
    n = int(min(len(high), len(low), len(close)))
    if period < 1 or n < period + 1:
        return 0.0
    h = np.asarray(high[n - period - 1 :], dtype=np.float64)
    l = np.asarray(low[n - period - 1 :], dtype=np.float64)
    c = np.asarray(close[n - period - 1 :], dtype=np.float64)
    prev_close = c[:-1]
    span = h[1:] - l[1:]
    up = np.abs(h[1:] - prev_close)
    down = np.abs(l[1:] - prev_close)
    true_range = np.maximum(np.maximum(span, up), down)
    atr = float(true_range.mean())
    if not math.isfinite(atr) or atr <= 0.0:
        return 0.0
    return atr


class M30BbDeepreturnOptimizedStrategy(BaseStrategy):
    """m30_bb_deepreturn_optimized —— 妙算导出的单因子策略。

    ★ 可调参数写成**类属性**（而非内联字面量），``tune`` 可直接改写并配合
    ``PARAM_SPACE`` 做搜索；改完记得同步 ``STRATEGY_VERSION`` 与 ``STRATEGY_CHANGELOG``。
    """

    # ── 身份 ───────────────────────────────────────────────────────────────
    #: **引擎加载主键**（docs/strategy_dev_guide.md:28）：必须等于
    #: ``settings.STRATEGY_POOL`` 的 key。引擎用 ``scan_strategies().get(name)``
    #: 查表（main.py:106 / main.py:529），而 scanner 取的是 ``cls.name``；
    #: 基类默认值是 ``"base"``，不声明就会被判为 Unknown strategy 直接跳过 ——
    #: 签名 / 继承 / magic 全对也照样不加载。
    name = "m30_bb_deepreturn_optimized"
    STRATEGY_MAGIC = STRATEGY_MAGIC
    STRATEGY_NAME = STRATEGY_NAME
    STRATEGY_VERSION = STRATEGY_VERSION
    STRATEGY_CHANGELOG = STRATEGY_CHANGELOG

    # ── ★ 可调参数（类属性，非内联字面量）─────────────────────────────────
    NEUTRAL_BAND = 0.05  # 中性带：|tanh(factor)| 小于该值视为不交易
    #: 预热根数：历史不足则不出信号（因子前 N 根为 0，无意义）。
    WARMUP_BARS = 499

    #: 因子滚动归一化窗口（与挖掘时一致，改动会导致 train-serve 不一致）。
    ROLL_WINDOW = 500

    #: 是否允许做空。
    LONG_SHORT = True

    #: 交易标的（按需改写 / 由外部注入）。
    SYMBOL = "XAUUSD"

    # ── ★ 风控参数（类属性，便于 tune / 人工复核）─────────────────────────
    #: 硬止损距离 = ``SL_ATR_MULT × ATR``。默认 3.0：比平台兜底的 2×ATR 更宽，
    #: 减少噪音扫损（用户拍板，对齐回测口径）。
    SL_ATR_MULT = 3.0
    #: 止盈距离 = ``TP_ATR_MULT × ATR``；``<= 0`` 表示**无固定止盈**
    #: （趋势跟踪，靠因子反向出场），与回测口径一致。
    TP_ATR_MULT = 0.0
    #: 硬止损最小距离（价格单位）：ATR 极小时防止止损贴脸被噪音扫掉。
    MIN_SL_POINTS = 3.0
    #: 止盈最小距离（价格单位）：ATR 极小时防止止盈贴脸。
    #: **仅在 TP_ATR_MULT > 0 时生效**（``take_dist = max(...) if tp_mult > 0.0
    #: else 0.0``）——TP 关闭时本值不参与任何计算。
    #: 重新启用止盈时需按所选倍数重推导本值：建议 =
    #: ``MIN_SL_POINTS × TP_ATR_MULT / SL_ATR_MULT`` 以保盈亏比（不得复用
    #: MIN_SL_POINTS，否则 ATR→0 时两侧地板相同，盈亏比被压成 1:1）。
    MIN_TP_POINTS = 0.0
    #: ATR 周期（与现网参考实现一致）。
    ATR_PERIOD = 14

    @property
    def required_bars(self) -> int:
        """策略启动所需的最少历史 K 线根数（预热 + 最长特征窗口余量）。"""
        return int(self.WARMUP_BARS) + 300

    def _current_atr(self) -> float:
        """用**已收盘** K 线算当前 ATR（bar1 语义：剔除正在形成的最后一根）。

        不依赖 DataFactory 指标缓存（``get_indicator`` 在部分部署下为空），
        直接从 ``self.candles`` 自算，因此与 ``generate_signal`` 用的是同一批
        数据 —— 盘中和收盘后再算，得到的 ATR 完全一致。

        Returns:
            ATR（价格单位）；数据不足时 ``0.0``。
        """
        candles = getattr(self, "candles", None)
        if not candles or len(candles) < 2:
            return 0.0
        arrays = _candles_to_arrays(candles[:-1])
        if arrays is None:
            return 0.0
        _open, high, low, close, _volume = arrays
        return _atr_from_arrays(high, low, close, int(self.ATR_PERIOD))

    def get_dynamic_sl_tp(
        self,
        direction: Any,
        entry_price: float,
        atr_val: float | None = None,
        position_type: str = "entry",
    ) -> tuple[float | None, float | None]:
        """按平台契约返回**绝对价格**形式的 (止损价, 止盈价)。

        平台调用契约（**以引擎实际代码为准，不是文档**）：
        ``main.py:1901/1903`` 与 ``main.py:1953`` 都只传 **2 个位置参数**
        （``direction`` + ``entry_price``），因此第 3/4 个参数**必须有默认值**，
        否则引擎一调用就 ``TypeError``，被 ``except`` 吞掉退化成固定点数 ——
        现网 ``20260819_goodma_v1`` 就是这样，动态 SL/TP 实际从未生效。

        返回值语义（读 ``main.py:1900-1956`` 与 ``athlete.py:110-115`` 确认）：
        **绝对价格**，不是距离、不是点数。引擎把返回值直接传进
        ``open_order(sl=…, tp=…)``；``athlete`` 侧 ``if not _sl or not _tp`` 与
        ``_execute_order`` 侧 ``if sl is None or tp is None`` 都会把"没给"当作
        回退信号。**注意**：``0.0`` 只对 athlete 算"没给"，对 ``_execute_order``
        却算"给了"——会用 ``sl=0`` 挂单。所以入场价非法时这里统一返回
        ``(None, None)``，**绝不返回 ``(0, 0)``**，让两条路径都走各自兜底。

        **TP 关闭的刻意语义**（``TP_ATR_MULT <= 0``，默认即如此）：止盈返回
        ``0.0`` 而不是 ``None`` —— 与 ``core.bridge.open_order(sl=0, tp=0)``
        的"未提供"默认语义一致，``tp=0`` 即不设固定止盈（趋势跟踪，持仓全靠
        因子反向平仓出场，与回测口径一致）。**只对 tp 用 0.0**；止损距离
        永远是真实价格，非法入场时仍是 ``(None, None)``。

        ATR 取值优先级（**bar1 优先**，与因子口径一致）：
        ``atr_val`` → 已收盘 K 线自算 :meth:`_current_atr` → 平台指标缓存
        （``get_indicator("atr")``）→ 百分比距离兜底。把自算排在缓存前面，是因为
        缓存 ATR 含**正在形成的最后一根**（EA / TA-Lib 均如此），而本策略的因子
        只用已收盘 K 线；风控与因子同源，盘中/收盘后取值一致、不重绘。

        方向归一（见 :func:`_is_buy_direction`）：**非买入**方向一律走卖出分支
        （``sl = price + stop_dist`` / ``tp = price - take_dist``）。引擎只会传
        ``OrderType`` 枚举（``BUY`` / ``SELL``），该兜底分支在正常调用下不可达；
        这里**刻意保持"非买即卖"的行为不变**，仅作说明 —— 这样将来平台新增枚举值
        也不会被误判成买入，最坏是当成卖出（方向确定，不会两头乱挂）。

        Args:
            direction: ``OrderType`` 枚举 / ``"BUY"`` 字符串 / 数值（见
                :func:`_is_buy_direction`）。
            entry_price: 入场价（买用 ask、卖用 bid，由引擎给定）。
            atr_val: 调用方传入的 ATR（新式签名用法）；``None`` 时自算。
            position_type: 平台预留位（``"entry"`` / 其他），本实现不使用。

        Returns:
            ``(sl_price, tp_price)``。买入方向 ``sl_price < entry < tp_price``，
            卖出方向 ``tp_price < entry < sl_price``（``TP_ATR_MULT<=0`` 时无固定
            止盈，``tp_price=0``）。**入场价非法**时返回 ``(None, None)``。
        """
        _ = position_type  # 平台预留参数，保留以免引擎按新签名调用时 TypeError
        try:
            price = float(entry_price)
        except (TypeError, ValueError):
            return None, None
        if not math.isfinite(price) or price <= 0.0:
            return None, None

        atr = 0.0
        if atr_val is not None:
            try:
                atr = float(atr_val)
            except (TypeError, ValueError):
                atr = 0.0
        if atr <= 0.0 or not math.isfinite(atr):
            # 优先自算（只吃已收盘 K 线，bar1 语义）。
            atr = self._current_atr()
        if atr <= 0.0 or not math.isfinite(atr):
            # 自算不可得时退到平台指标缓存（EA / TA-Lib 提供，含最后一根）。
            # 用 getattr 取，兼容未提供该便捷方法的基类（离线占位 / 其他平台）。
            _get_indicator = getattr(self, "get_indicator", None)
            cached = _get_indicator("atr") if callable(_get_indicator) else None
            try:
                atr = float(cached) if cached is not None else 0.0
            except (TypeError, ValueError):
                atr = 0.0
        if atr <= 0.0 or not math.isfinite(atr):
            # ATR 完全不可得：退化成百分比距离。宁可止损离得远，也不给 0
            # （0 会被 athlete 判为"未提供"而改用硬编码 ATR=15）。
            atr = price * 0.005  # 0.5% —— 与 3×ATR 在 ATR≈price/600 时同量级

        stop_dist = max(atr * float(self.SL_ATR_MULT), float(self.MIN_SL_POINTS))
        tp_mult = float(self.TP_ATR_MULT)
        # 止盈地板用**独立的** MIN_TP_POINTS（不是 MIN_SL_POINTS），否则 ATR 极小时
        # 止损/止盈贴脸同距、盈亏比被压成 1:1（仅 TP_ATR_MULT > 0 时相关；
        # TP 关闭时 take_dist 恒为 0，本值不参与计算）。
        take_dist = max(atr * tp_mult, float(self.MIN_TP_POINTS)) if tp_mult > 0.0 else 0.0

        if _is_buy_direction(direction):
            return price - stop_dist, (price + take_dist if take_dist > 0.0 else 0.0)
        return price + stop_dist, (price - take_dist if take_dist > 0.0 else 0.0)

    def generate_signal(self) -> tuple[Any, ...] | None:
        """按 妙算 平台契约生成交易信号（**无参**；K 线从 ``self.candles`` 读）。

        平台契约（``strategies.base.BaseStrategy``）：

        * 基类 ``on_tick`` 先 ``refresh_data()`` 把 K 线灌进 ``self.candles``，
          再以**无参**方式调用本方法。因此这里**不能**声明 ``candles`` 形参 ——
          否则 妙算 一调用就 ``TypeError``，导出策略根本加载不起来。
        * 返回 ``None``（无信号）或六元组
          ``(signal, score_long, score_short, factors_long, factors_short, indicator_values)``；
          ``signal`` 必须是 ``OrderType`` 枚举（``on_tick`` 会读 ``signal.value``）。

        bar1 语义（不重绘）：``self.candles[-1]`` 是**正在形成**的 K 线，本方法用
        ``self.candles[:-1]`` 把它剔除，只用已收盘数据计算，因此不存在未来函数。

        Returns:
            六元组；数据不足 / 因子中性 / 未过预热期时返回 ``None``。
        """
        candles = getattr(self, "candles", None)
        closed = candles[:-1] if candles is not None and len(candles) > 1 else candles
        arrays = _candles_to_arrays(closed)
        if arrays is None:
            return None
        factor = compute_factor(*arrays, float(getattr(self, "GATE_DEADBAND", 0.0)))
        if factor.size < int(self.WARMUP_BARS) + 1:
            return None
        value = float(factor[factor.size - 1])
        exposure = float(np.tanh(value))
        if abs(exposure) < float(self.NEUTRAL_BAND):
            return None
        if not self.LONG_SHORT and exposure < 0.0:
            return None
        score = int(round(min(abs(exposure), 1.0) * 100))
        # 真实 ATR（同一批已收盘 K 线）：供 get_dynamic_sl_tp 与平台兜底共用，
        # 避免 athlete.py:113 回退到硬编码的 15。缺失则记 None，由风控层兜底。
        atr_value = _atr_from_arrays(*arrays[1:4], int(self.ATR_PERIOD))
        indicator_values: dict[str, Any] = {
            "symbol": self.SYMBOL,
            "magic": self.STRATEGY_MAGIC,
            "factor": round(value, 4),
            "exposure": round(exposure, 4),
            "bars": int(factor.size),
            "atr": round(atr_value, 6) if atr_value > 0.0 else None,
        }
        if exposure > 0.0:
            return (OrderType.BUY, score, 0, [f"{self.STRATEGY_NAME}-LONG"], [], indicator_values)
        return (OrderType.SELL, 0, score, [], [f"{self.STRATEGY_NAME}-SHORT"], indicator_values)
