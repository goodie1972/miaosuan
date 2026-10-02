# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""策略说明文档生成器（神机格式，中英双语）+ spec↔策略映射表。

目录职责（重构后，2026-10-01）：

* ``artifacts/``   —— **spec 专属**：只放 StrategySpec JSON 及其逐代 sidecar；
* ``strategies/``  —— **策略专属**：策略 .py + ``<名>_README_zh.md`` /
  ``<名>_README_en.md`` 双语说明 + ``spec_strategy_map.json`` 映射表；
* ``data/results/``—— spec 关联文档：回测 / 寻优结果（reference-only）；
* ``data/cache/``  —— 唯一缓存目录（所有 OHLCV parquet/csv 缓存）；
* ``tmp/``         —— 中间 / 临时文件（可安全清空）。

本模块提供：

* :func:`generate_readme_zh` / :func:`generate_readme_en` —— 按神机文档格式
  （公式 RPN → 语义 → 风控 → 证据 → 溯源）生成 Markdown 说明文本；
* :func:`update_strategy_map` —— 维护 ``strategies/spec_strategy_map.json``，
  记录 策略文件 ↔ spec 文件 的双向映射（spec_id 为关联主键）；
* :func:`write_strategy_docs` —— export 流程的唯一入口：写双语 README +
  更新映射表，返回全部产出路径。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .ir.schema import StrategySpec


def _coerce_spec(spec: Any) -> Any:
    """如果 *spec* 是 dict，用 ``StrategySpec.from_dict`` 重建；否则原样返回。

    这样 :func:`write_strategy_docs` 同时接受真正的 StrategySpec 对象和
    从 JSON 文件加载的纯 dict。
    """
    if isinstance(spec, dict):
        from .ir.schema import StrategySpec as _SS
        return _SS.from_dict(spec)
    return spec


def _param_to_dict(p: Any) -> dict[str, Any]:
    """安全把 param_space 元素转成 dict。"""
    if hasattr(p, "to_dict"):
        return p.to_dict()
    if isinstance(p, dict):
        return p
    # 字符串或其他不可迭代类型 → 包装成占位行
    if isinstance(p, str):
        return {"name": p, "default": "—", "kind": "—", "description": ""}
    return {"name": str(p), "default": "—", "kind": "—", "description": ""}

__all__ = [
    "MAP_FILENAME",
    "generate_readme_en",
    "generate_readme_zh",
    "update_strategy_map",
    "write_strategy_docs",
]

#: 映射表文件名（放在 strategies目录根）。
MAP_FILENAME = "spec_strategy_map.json"


# ── 文本工具 ────────────────────────────────────────────────────────────────


def _fmt(value: Any, digits: int = 4) -> str:
    """安全格式化：None → "—"；浮点 → 定长小数；其他 → str。"""
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _evidence_rows(evidence: Any) -> tuple[str, str, str, str, str, str]:
    """提取证据六件套的展示串（中英文档共用取值，只是表头不同）。

    兼容 dataclass 对象和 dict：对 dict 使用 ``deployable = gate_verdict == "DEPLOYABLE"``。
    """
    if isinstance(evidence, dict):
        val = evidence.get("val_score")
        folds = evidence.get("wf_folds", 0)
        dsr = evidence.get("deflated_sharpe")
        trials = evidence.get("n_trials", 0)
        verdict = str(evidence.get("gate_verdict", "UNKNOWN"))
        deploy = verdict == "DEPLOYABLE"
    else:
        val = evidence.val_score
        folds = evidence.wf_folds
        dsr = evidence.deflated_sharpe
        trials = evidence.n_trials
        verdict = str(evidence.gate_verdict)
        deploy = evidence.deployable
    return (
        _fmt(val),
        str(folds),
        _fmt(dsr),
        str(trials),
        "PASS" if deploy else "FAIL",
        verdict,
    )


# ── 中文版 ──────────────────────────────────────────────────────────────────


def generate_readme_zh(
    spec: StrategySpec,
    *,
    strategy_file: str,
    magic: int | None,
    formula: str,
    param_space: list[dict[str, Any]] | tuple[Any, ...] = (),
) -> str:
    """按神机文档格式生成中文说明（Markdown）。

    Args:
        spec: 策略的 StrategySpec。
        strategy_file: 同目录策略文件名（如 ``20261001_x_v1.py``）。
        magic: 分配到的 magic（可为 None）。
        formula: 可读公式串（``readable_formula`` 输出）。
        param_space: 参数空间（``ParamSpace.to_dict()`` 结果或对象列表）。
    """
    ev = spec.evidence
    pv = spec.provenance
    se = spec.semantics
    val_s, folds, dsr, trials, deploy_s, verdict = _evidence_rows(ev)
    reasons = "\n".join(f"  - {r}" for r in ev.gate_reasons) or "  - （无）"
    params_md = ["| 参数 | 默认值 | 类型 | 说明 |", "|---|---|---|---|"]
    for p in param_space:
        d = _param_to_dict(p)
        params_md.append(
            f"| `{d.get('name', '')}` | {d.get('default', '—')} | "
            f"{d.get('kind', '—')} | {d.get('description', '')} |"
        )
    param_table = "\n".join(params_md) if len(params_md) > 2 else "（无可调参数）"

    return f"""# {spec.name} —— 策略说明

> 由妙算（MiaoSuan）自动生成，**请勿手工编辑**。
> 策略文件：`{strategy_file}` ｜ 门禁结论：**{verdict}** ｜ {deploy_s}

## 一、公式（RPN）

```
{formula}
```

共 {len(spec.payload.tokens)} 个 token，词表版本 `{pv.vocab_version or getattr(spec.payload, "vocab_version", "")}`
（token 仅在同一词表下有意义，换词表必须重新导出）。

## 二、语义

| 项目 | 取值 | 说明 |
|---|---|---|
| 滚动窗口 | {se.roll_window} | z-score 窗口，前 {se.roll_window - 1} 根输出 0（clip ±3） |
| 预热 bar | {se.warmup_bars} | 起算前丢弃的 bar 数 |
| 中性带 | {se.neutral_band} | \\|tanh(因子)\\| 低于该值不出信号 |
| 多空 | {"双向" if se.long_short else "单向"} | 仓位 = tanh(因子)，与回测口径一致 |
| bar1 语义 | 不重绘 | 只用已收盘 K 线，最后一根未收盘 bar 不参与计算 |

## 三、风控

* 硬止损：入场价 ∓ 3.0 × ATR(14)（ATR 用收盘 K 线真实值，非平台兜底常量 15）。
* 止盈：默认**无固定止盈**（趋势跟踪，因子反向出场），与回测口径一致。

## 四、可调参数

{param_table}

## 五、证据（导出时刻快照，只读）

| 指标 | 取值 |
|---|---|
| 验证集得分 | {val_s} |
| Walk-Forward 折数 | {folds} |
| 去膨胀夏普 DSR | {dsr} |
| 试验次数 | {trials} |
| 成本敏感度 | {_cost_line(ev)} |
| hold-out 夏普 | {_holdout_line(ev)} |
| 门禁结论 | {verdict}（{deploy_s}） |

门禁未通过原因：
{reasons}

## 六、溯源

| 项目 | 取值 |
|---|---|
| IR spec_id | `{spec.spec_id}` |
| 生成时刻 | {pv.created_at or "—"} |
| git sha | `{pv.git_sha or "—"}` |
| 数据指纹 | `{pv.data_fingerprint or "—"}` |
| 随机种子 | {pv.seed if pv.seed is not None else "—"} |
| 市场/品种 | {pv.market or "—"} |
| 预算档位 | {pv.budget or "—"} |
| magic | {magic if magic is not None else "—"} |
| 对应 spec 文件 | 见 `spec_strategy_map.json` |

---
*本文件由 `miaosuan.strategy_docs` 自动生成；中文文档与英文文档（`*_README_en.md`）内容同源，*
*语义冲突时以本中文版为准，以 spec JSON 为最终裁决。*
"""


def _cost_line(evidence: Any) -> str:
    cs = getattr(evidence, "cost_sensitivity", None)
    if not cs:
        return "—"
    if isinstance(cs, dict):
        return "，".join(f"{k}={_fmt(v, 4)}" for k, v in cs.items())
    return str(cs)


def _holdout_line(evidence: Any) -> str:
    h = getattr(evidence, "holdout_sharpe", None)
    return _fmt(h) if h is not None else "未消费阅读本文前hold-out"


# ── 英文版 ──────────────────────────────────────────────────────────────────


def generate_readme_en(
    spec: StrategySpec,
    *,
    strategy_file: str,
    magic: int | None,
    formula: str,
    param_space: list[dict[str, Any]] | tuple[Any, ...] = (),
) -> str:
    """Generate the English README (mirror of :func:`generate_readme_zh`)."""
    ev = spec.evidence
    pv = spec.provenance
    se = spec.semantics
    val_s, folds, dsr, trials, deploy_s, verdict = _evidence_rows(ev)
    reasons = "\n".join(f"  - {r}" for r in ev.gate_reasons) or "  - (none)"
    params_md = ["| Param | Default | Kind | Description |", "|---|---|---|---|"]
    for p in param_space:
        d = _param_to_dict(p)
        params_md.append(
            f"| `{d.get('name', '')}` | {d.get('default', '—')} | "
            f"{d.get('kind', '—')} | {d.get('description', '')} |"
        )
    param_table = "\n".join(params_md) if len(params_md) > 2 else "(no tunable parameters)"

    return f"""# {spec.name} — Strategy README

> Auto-generated by MiaoSuan. **Do not edit by hand.**
> Strategy file: `{strategy_file}` | Gate verdict: **{verdict}** | {deploy_s}

## 1. Formula (RPN)

```
{formula}
```

{len(spec.payload.tokens)} tokens, vocab version `{pv.vocab_version or getattr(spec.payload, "vocab_version", "")}`
(tokens are only meaningful under the same vocab; re-export on vocab change).

## 2. Semantics

| Item | Value | Notes |
|---|---|---|
| Roll window | {se.roll_window} | z-score window; first {se.roll_window - 1} bars output 0 (clip ±3) |
| Warmup bars | {se.warmup_bars} | Bars discarded before scoring starts |
| Neutral band | {se.neutral_band} | \\|tanh(factor)\\| below this emits no signal |
| Long/short | {"both" if se.long_short else "single side"} | exposure = tanh(factor), matches backtest |
| bar1 semantics | non-repainting | Uses closed candles only; last unclosed bar is ignored |

## 3. Risk control

* Hard stop: entry ∓ 3.0 × ATR(14) (true ATR from closed candles, not the platform's constant 15).
* Take-profit: none by default (trend-following; exit on factor flip), matching the backtest definition.

## 4. Tunable parameters

{param_table}

## 5. Evidence (point-in-time snapshot at export, read-only)

| Metric | Value |
|---|---|
| Validation score | {val_s} |
| Walk-forward folds | {folds} |
| Deflated Sharpe (DSR) | {dsr} |
| Number of trials | {trials} |
| Cost sensitivity | {_cost_line(ev)} |
| Hold-out Sharpe | {_holdout_line_en(ev)} |
| Gate verdict | {verdict} ({deploy_s}) |

Gate failure reasons:
{reasons}

## 6. Provenance

| Item | Value |
|---|---|
| IR spec_id | `{spec.spec_id}` |
| Generated at | {pv.created_at or "—"} |
| git sha | `{pv.git_sha or "—"}` |
| Data fingerprint | `{pv.data_fingerprint or "—"}` |
| Random seed | {pv.seed if pv.seed is not None else "—"} |
| Market / symbol | {pv.market or "—"} |
| Budget | {pv.budget or "—"} |
| magic | {magic if magic is not None else "—"} |
| Spec file | see `spec_strategy_map.json` |

---
*Auto-generated by `miaosuan.strategy_docs`. The Chinese README (`*_README_zh.md`) is*
*content-equivalent; on any discrepancy the Chinese version prevails, with the JSON*
*spec as the final source of truth.*
"""


def _holdout_line_en(evidence: Any) -> str:
    h = getattr(evidence, "holdout_sharpe", None)
    return _fmt(h) if h is not None else "hold-out not consumed"


# ── 映射表 ─────────────────────────────────────────────────────────────


def update_strategy_map(
    strategies_dir: Path | str,
    entry: dict[str, Any],
) -> Path:
    """把一行映射写入/更新到 ``strategies/spec_strategy_map.json``。

    Structure of the map file::

        {
          "version": 1,
          "updated_at": "<ISO8601>",
          "strategies": [ { ...entry... }, ... ]
        }

    合并规则：以 ``strategy_file`` 为键覆盖同一条目（重复导出更新而非追加），
    按导出时间正序保存。返回映射表Path。
    """
    directory = Path(strategies_dir)
    directory.mkdir(parents=True, exist_ok=True)
    map_path = directory / MAP_FILENAME

    data: dict[str, Any] = {"version": 1, "updated_at": "", "strategies": []}
    if map_path.is_file():
        try:
            loaded = json.loads(map_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and isinstance(loaded.get("strategies"), list):
                data = loaded
        except (OSError, ValueError):
            pass  # 损坏则重建（映射是索引，不是不可再生数据）

    strategies = [s for s in data["strategies"] if s.get("strategy_file") != entry.get("strategy_file")]
    strategies.append(entry)
    strategies.sort(key=lambda s: str(s.get("generated_at", "")))

    data["strategies"] = strategies
    data["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())
    map_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return map_path


# ── export 流程入口 ─────────────────────────────────────────────────────────


def write_strategy_docs(
    spec: StrategySpec,
    *,
    strategy_path: Path | str,
    magic: int | None,
    formula: str,
    param_space: list[dict[str, Any]] | tuple[Any, ...] = (),
    spec_file: str = "",
) -> list[Path]:
    """写双语 README + 更新映射表（export 成功后的唯一入口）。

    Args:
        spec: 策略 spec（可以是 :class:`StrategySpec` 对象或等价的 dict）。
        strategy_path: 刚写盘的策略 .py 全路径（其所在目录即 strategies 目录）。
        magic: 分配的 magic。
        formula: 可读公式串。
        param_space: 参数空间。
        spec_file: 触发导出的 spec JSON 文件名（写入映射表，可为空）。

    Returns:
        新生成 / 更新的全部文件路径（两个 README + 映射表）。
    """
    spec = _coerce_spec(spec)  # dict → StrategySpec
    target = Path(strategy_path)
    out_dir = target.parent
    stem = target.stem

    kwargs = dict(
        strategy_file=target.name, magic=magic, formula=formula, param_space=param_space
    )
    zh_path = out_dir / f"{stem}_README_zh.md"
    en_path = out_dir / f"{stem}_README_en.md"
    zh_path.write_text(generate_readme_zh(spec, **kwargs), encoding="utf-8")
    en_path.write_text(generate_readme_en(spec, **kwargs), encoding="utf-8")

    map_path = update_strategy_map(
        out_dir,
        {
            "strategy_file": target.name,
            "spec_file": spec_file,
            "spec_id": spec.spec_id,
            "name": spec.name,
            "magic": magic,
            "vocab_version": spec.provenance.vocab_version
            or getattr(spec.payload, "vocab_version", ""),
            "market": spec.provenance.market,
            "generated_at": spec.provenance.created_at
            or time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
            "gate_verdict": spec.evidence.gate_verdict,
            "deployable": bool(spec.evidence.deployable),
            "docs": {"zh": zh_path.name, "en": en_path.name},
        },
    )
    return [zh_path, en_path, map_path]
