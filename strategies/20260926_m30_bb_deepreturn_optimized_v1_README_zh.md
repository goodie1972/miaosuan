# m30_bb_deepreturn_optimized —— 策略说明

> 由妙算（MiaoSuan）自动生成，**请勿手工编辑**。
> 策略文件：`20260926_m30_bb_deepreturn_optimized_v1.py` ｜ 门禁结论：**RESEARCH_ONLY** ｜ FAIL

## 一、公式（RPN）

```
-(BOLL_POS + MFI14)
```

共 4 个 token，词表版本 `v9217a2c0d91a`
（token 仅在同一词表下有意义，换词表必须重新导出）。

## 二、语义

| 项目 | 取值 | 说明 |
|---|---|---|
| 滚动窗口 | 500 | z-score 窗口，前 499 根输出 0（clip ±3） |
| 预热 bar | 499 | 起算前丢弃的 bar 数 |
| 中性带 | 0.05 | \|tanh(因子)\| 低于该值不出信号 |
| 多空 | 双向 | 仓位 = tanh(因子)，与回测口径一致 |
| bar1 语义 | 不重绘 | 只用已收盘 K 线，最后一根未收盘 bar 不参与计算 |

## 三、风控

* 硬止损：入场价 ∓ 3.0 × ATR(14)（ATR 用收盘 K 线真实值，非平台兜底常量 15）。
* 止盈：默认**无固定止盈**（趋势跟踪，因子反向出场），与回测口径一致。

## 四、可调参数

（无可调参数）

## 五、证据（导出时刻快照，只读）

| 指标 | 取值 |
|---|---|
| 验证集得分 | — |
| Walk-Forward 折数 | 0 |
| 去膨胀夏普 DSR | — |
| 试验次数 | 1 |
| 成本敏感度 | — |
| hold-out 夏普 | 未消费阅读本文前hold-out |
| 门禁结论 | RESEARCH_ONLY（FAIL） |

门禁未通过原因：
  - LLM-assisted conversion from Shenji strategy 20260711_m30_bb_deepreturn_optimized_v1.py
  - Entry factor only; exit logic delegated to composition layer
  - Signal overlap verified: 100% match at original entry points (131 long + 105 short)
  - Trend confirmation + ATR volatility bonus delegated to composition layer

## 六、溯源

| 项目 | 取值 |
|---|---|
| IR spec_id | `1e01feae75453529` |
| 生成时刻 | 2026-09-26T06:29:43.094052+00:00 |
| git sha | `2277081` |
| 数据指纹 | `—` |
| 随机种子 | 0 |
| 市场/品种 | FOREX_XAUUSD |
| 预算档位 | manual |
| magic | 661102 |
| 对应 spec 文件 | 见 `spec_strategy_map.json` |

---
*本文件由 `miaosuan.strategy_docs` 自动生成；中文文档与英文文档（`*_README_en.md`）内容同源，*
*语义冲突时以本中文版为准，以 spec JSON 为最终裁决。*
