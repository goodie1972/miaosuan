"""特征注册（M2 脚手架 —— 仅注册名称与类别，**不**含 numpy 实现）。

.. warning::

   本文件在 T01/M2 阶段是**过渡脚手架**。它的唯一职责是：以与冻结 AM
   （``model_core/features.py`` 的 ``_FEATURE_DEFS``）**逐元素一致**的名称与顺序
   注册 65 个特征，从而让 :mod:`miaosuan.core.vocab` 能派生出于 AM **逐字符相同**的
   ``VOCAB_VERSION``（== ``v9217a2c0d91a``）。

   **M4 将把本文件替换为 numpy 化实现**（填充 ``compute`` 的真实张量计算逻辑）。
   届时**必须保持** ``_FEATURE_DEFS`` 的名称与顺序**一字不改**，否则 VOCAB_VERSION
   会漂移、旧产物将被 ``verify()`` 拒绝。

特征顺序严格对应 AM ``compute_features`` 的 stack 顺序（0..64），不可变更。
每条 ``compute`` 当前为占位函数，调用即抛 ``NotImplementedError``（显式失败，
绝不静默返回错误结果 —— 规避 AM 的 ``except: pass`` 反模式）。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .registry import FeatureSpec, Registry

__all__ = ["FEATURE_REGISTRY", "FEATURE_NAMES", "FEATURE_COUNT", "is_registered"]


def _pending_feature(name: str) -> Callable[[dict[str, Any]], Any]:
    """构造占位 compute 函数（M4 前调用即显式失败）。"""

    def _compute(raw: dict[str, Any]) -> Any:  # noqa: ARG001 - 占位实现忽略入参
        raise NotImplementedError(
            f"特征 '{name}' 的 numpy 计算实现将在 M4 提供；当前仅为词表脚手架。"
        )

    _compute.__name__ = f"_pending_{name}"
    return _compute


# ── 65 个特征的 (name, category) ─────────────────────────────────────────
# 顺序即 token/特征维顺序，逐项对齐 AM model_core/features.py::_FEATURE_DEFS。
_FEATURE_DEFS: list[tuple[str, str]] = [
    # 趋势类 trend (0-4)
    ("RET", "trend"),
    ("RET5", "trend"),
    ("RET20", "trend"),
    ("MA_DIFF", "trend"),
    ("SLOPE20", "trend"),
    # 波动类 volatility (5-8)
    ("ATR", "volatility"),
    ("RVOL", "volatility"),
    ("HL_RANGE", "volatility"),
    ("VOL_REGIME", "volatility"),
    # 反转类 reversal (9-13)
    ("DEV", "reversal"),
    ("DEV60", "reversal"),
    ("RSI14", "reversal"),
    ("PRESSURE", "reversal"),
    ("AC1", "reversal"),
    # 成交量类 volume (14-16)
    ("VOL_RATIO", "volume"),
    ("VOL_Z", "volume"),
    ("PV_CORR", "volume"),
    # 跨截面相对强弱 cross_sectional (17-19)
    ("REL_RET5", "cross_sectional"),
    ("REL_RET20", "cross_sectional"),
    ("REL_VOL", "cross_sectional"),
    # v3.0 新增特征 (20-25)
    ("VWAP_DEV", "volume"),
    ("BOLL_POS", "channel"),
    ("BOLL_WIDTH", "volatility"),
    ("MACD_HIST", "momentum"),
    ("OBV_SLOPE", "volume"),
    ("MFI14", "volume"),
    # v3.0 Alpha 101 + 互补特征 (26-29)
    ("WILLR_14", "reversal"),
    ("CCI_14", "reversal"),
    ("ROC_12", "momentum"),
    ("TYPICAL_DEV", "reversal"),
    # task 5.2 趋势类 trend (30-32)
    ("EMA_RATIO_12_26", "trend"),
    ("TREND_STRENGTH_50", "trend"),
    ("PRICE_POS_50", "trend"),
    # task 5.2 动量类 momentum (33-36)
    ("TRIX_15", "momentum"),
    ("PPO", "momentum"),
    ("ULT_OSC", "momentum"),
    ("RET_ACCEL", "momentum"),
    # task 5.3 波动类（含 OHLC 估计量）volatility (37-40)
    ("GK_VOL", "volatility"),
    ("PARKINSON_VOL", "volatility"),
    ("YANG_ZHANG_VOL", "volatility"),
    ("RS_VOL", "volatility"),
    # task 5.4 量能/流动性类 volume (41-44)
    ("AMIHUD_ILLIQ", "volume"),
    ("KYLE_LAMBDA", "volume"),
    ("CMF_20", "volume"),
    ("AD_LINE_SLOPE", "volume"),
    # task 5.5 反转/振荡类 reversal/trend/momentum (45-50)
    ("STOCH_K_14", "reversal"),
    ("STOCH_D_3", "reversal"),
    ("AROON_OSC_25", "reversal"),
    ("DMI_ADX_14", "trend"),
    ("DMI_DIFF_14", "trend"),
    ("TRIX_SIGNAL", "momentum"),
    # task 5.6 通道/突破类 channel (51-56)
    ("DONCHIAN_POS_20", "channel"),
    ("KELTNER_POS_20", "channel"),
    ("ICHIMOKU_KIJUN_DEV", "channel"),
    ("ICHIMOKU_TENKAN_DEV", "channel"),
    ("SUPERTREND_DIR", "channel"),
    ("SAR_DIST", "channel"),
    # task 5.7 统计类 statistical (57-62)
    ("ROLL_SKEW_20", "statistical"),
    ("ROLL_KURT_20", "statistical"),
    ("HURST_50", "statistical"),
    ("FRACTAL_DIM_30", "statistical"),
    ("AC2", "statistical"),
    ("RET_ENTROPY_20", "statistical"),
    # task 5.8 跨截面相对强弱补充 cross_sectional (63-64)
    ("CS_RANK_RET5", "cross_sectional"),
    ("CS_ZSCORE_RET20", "cross_sectional"),
]


# ── 构建 FEATURE_REGISTRY ────────────────────────────────────────────────

FEATURE_REGISTRY = Registry()

for _name, _category in _FEATURE_DEFS:
    FEATURE_REGISTRY.register_feature(
        FeatureSpec(name=_name, category=_category, compute=_pending_feature(_name))
    )

# 由注册表导出有序特征名视图（保持下游 import 兼容）
FEATURE_NAMES: tuple[str, ...] = FEATURE_REGISTRY.feature_names

# 特征总数（= 65，与 AM 一致）
FEATURE_COUNT: int = len(FEATURE_REGISTRY.feature_names)


def is_registered(name: str) -> bool:
    """该 token 名称是否已注册（跨 Feature/Operator 全局唯一）。"""
    return name in FEATURE_REGISTRY
