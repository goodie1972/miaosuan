# Prompt 2: 指标 → 妙算 Token 映射

## 角色
你是妙算词表专家。请将策略用到的指标映射到妙算可用 token，标注缺口。

## 输入
- 策略结构化摘要（Prompt 1 输出 JSON）
- 妙算词表导出（见 `vocab_export.json`，完整特征/算子名单）

## 输出格式（严格 JSON）
```json
{
  "mapping": {
    "策略指标名": "妙算 token 名|null",
    "bb_upper": "BB_UPPER",
    "bb_lower": "BB_LOWER",
    "bb_mid": null,
    "mfi": "MFI14",
    "atr": "ATR",
    "adx": null
  },
  "missing_indicators": [
    {"indicator": "adx", "reason": "词表无 ADX", "alternatives": ["DMI_ADX_14", "TREND_STRENGTH_50"], "notes": "DMI_ADX_14 更接近标准 ADX 定义"}
  ],
  "partial_mapping": [
    {"indicator": "bb_width", "mapped_to": "BOLL_WIDTH", "notes": "仅映射宽度，中轨无对应 token"}
  ],
  "multi_token_suggestions": [
    {"indicator": "mfi_oversold_cross", "suggestion": "用 MFI14 + IF_GT/GATE 组合表达 MFI 穿越超卖线"}
  ],
  "rpn_feasibility": "high|medium|low"
}
```

## 映射原则
1. **精确匹配优先**：策略指标名直接对应词表特征名（如 `MFI14`、`ATR`、`BB_UPPER`、`BOLL_WIDTH`）
2. **近似匹配次之**：如 ADX → `DMI_ADX_14`，需在 notes 说明差异
3. **组合表达兜底**：单 token 无对应时，建议用多 token 组合（如超买超卖线穿越 = 特征 + 算子）
3. **标注不可映射**：明确列出完全无对应的指标，供研究员决策（舍弃/自定义/外部计算）

## 词表参考（仅部分，完整见 vocab_export.json）
- 特征（65 个）：RET/RET5/RET20/MA_DIFF/SLOPE20/ATR/RVOL/HL_RANGE/VOL_REGIME/DEV/DEV60/RSI14/PRESSURE/AC1/VOL_RATIO/VOL_Z/PV_CORR/REL_RET5/REL_RET20/REL_VOL/VWAP_DEV/BOLL_POS/BOLL_WIDTH/MACD_HIST/OBV_SLOPE/MFI14/WILLR_14/CCI_14/ROC_12/TYPICAL_DEV/EMA_RATIO_12_26/TREND_STRENGTH_50/PRICE_POS_50/TRIX_15/PPO/ULT_OSC/RET_ACCEL/GK_VOL/PARKINSON_VOL/YANG_ZHANG_VOL/RS_VOL/AMIHUD_ILLIQ/KYLE_LAMBDA/CMF_20/AD_LINE_SLOPE/STOCH_K_14/STOCH_D_3/AROON_OSC_25/DMI_ADX_14/DMI_DIFF_14/TRIX_SIGNAL/DONCHIAN_POS_20/KELTNER_POS_20/ICHIMOKU_KIJUN_DEV/ICHIMOKU_TENKAN_DEV/SUPERTREND_DIR/SAR_DIST/ROLL_SKEW_20/...
- 算子（62 个）：ADD/SUB/MUL/DIV/NEG/ABS/SIGN/GATE/JUMP/DECAY/WMA/DELTA/SCALE/CS_RANK/CS_SCALE/CS_NEUTRALIZE/MIN/MAX/POWER/SIGNED_LOG/SQRT/WINSORIZE/CLIP/SIGMOID/TANH_SQUASH/IF_GT/...

## 禁止事项
- 不要编造不存在的 token
- 不要输出非 JSON 文本