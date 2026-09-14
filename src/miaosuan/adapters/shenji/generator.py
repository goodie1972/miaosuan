# -*- coding: utf-8 -*-
# Copyright (c) 2026 Original Author
# Copyright (c) 2026 Your Company
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""神机 导出器（M15）：``StrategySpec`` → 神机 策略 ``.py``。

职责边界：

* 本模块**不写盘**：渲染结果经 :class:`~miaosuan.adapters.base.ExportResult`
  回传，写盘由 :mod:`miaosuan.cli` 完成；
* magic 号段分配走 :mod:`magic_registry` 账本（持久化状态，非临时文件）；
* 因子内核源码由 :mod:`kernel` 从 ``core`` 原样导出，保证数值保真；
* 渲染后立刻跑 :mod:`lint`，把 repaint / 常量缺失等问题挡在导出环节。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from ...core.vocab import FORMULA_VOCAB
from ...ir.schema import FactorPayload, StrategySpec
from ..base import ExportResult, LintIssue, ParamSpace, TargetPort
from .contract import GATE_TOKEN_NAME, PLATFORM_SPEC, default_filename
from .kernel import KernelPlan, build_kernel_plan
from .lint import lint_source
from .magic_registry import allocate_magic

__all__ = [
    "ATR_PERIOD",
    "ShenjiPort",
    "DEFAULT_TEMPLATE",
    "MIN_SL_POINTS",
    "MIN_TP_POINTS",
    "SL_ATR_MULT",
    "TP_ATR_MULT",
    "build_param_space",
    "readable_formula",
]

#: 默认模板文件名。
DEFAULT_TEMPLATE: str = "factor_kernel_v1.py.j2"

#: 模板目录（与包同目录下的 ``templates/``）。
_TEMPLATE_DIR: Path = Path(__file__).resolve().parent / "templates"

# ── 导出策略的默认风控参数（渲染成可调类属性）────────────────────────────────
#: 硬止损距离 = ``SL_ATR_MULT × ATR``。取 **3.0**（用户拍板）：比平台自身兜底
#: （``athlete.py:114-115``：``price ∓ atr*2``）更宽，减少噪音扫损。
#: 注意与平台兜底**不再同比例**——我们的 ATR 是真实值（已收盘 K 线自算），
#: 即使兜底路径被触发，止损位置也仅是平台自己的回退，不影响我们的单子。
SL_ATR_MULT: float = 3.0

#: 止盈距离 = ``TP_ATR_MULT × ATR``。取 **0.0 = 无固定止盈**（趋势跟踪：
#: 持仓全靠因子反向平仓出场）。与回测口径一致——回测层（``report/equity.py``）
#: 没有 SL/TP，设固定止盈反而会造成实盘与回测行为背离。
#: ``0`` 是 ``core.bridge.open_order(sl=0, tp=0)`` 的"未提供"默认语义，
#: **不要改成 ``None``**（``None`` 会绕过 athlete 的取值链）。
TP_ATR_MULT: float = 0.0

#: 硬止损最小距离（价格单位）：ATR 极小时防止止损贴脸被噪音扫掉。
#: 取自现网参考实现 ``strategies/20260909_h1_alphagate_v1.MIN_SL_POINTS``。
MIN_SL_POINTS: float = 3.0

#: 止盈最小距离（价格单位）：ATR 极小时防止止盈贴脸。
#: **仅在 ``TP_ATR_MULT > 0`` 时生效**（``take_dist = max(...) if tp_mult > 0.0
#: else 0.0``）——TP 默认关闭后本值不参与任何计算，推导结果为 ``0.0``。
#: **重新启用止盈时需按所选倍数重推导本值**（建议 =
#: ``MIN_SL_POINTS × TP_ATR_MULT / SL_ATR_MULT`` 以保盈亏比，且不得复用
#: ``MIN_SL_POINTS``——否则 ATR→0 时两侧地板相同，盈亏比被压成 1:1）。
MIN_TP_POINTS: float = (
    MIN_SL_POINTS * TP_ATR_MULT / SL_ATR_MULT if SL_ATR_MULT > 0.0 else MIN_SL_POINTS
)

#: ATR 周期。14 与现网 ``alphagate._get_atr`` / TA-Lib ``ATR`` 默认一致。
ATR_PERIOD: int = 14


def readable_formula(tokens: Sequence[int]) -> str:
    """把 token 序列渲染为人类可读的公式串。

    Args:
        tokens: RPN token 序列。

    Returns:
        形如 ``TRIX_15 → RET_ENTROPY_20 → MA_DIFF → MOMENTUM_5`` 的字符串。
    """
    names = FORMULA_VOCAB.token_names
    parts: list[str] = []
    for token in tokens:
        index = int(token)
        parts.append(names[index] if 0 <= index < len(names) else f"op_{index}")
    return " → ".join(parts)


def build_param_space(
    *,
    neutral_band: float,
    gate_deadband: float,
    has_gate: bool,
) -> tuple[ParamSpace, ...]:
    """构造默认可调参数空间（与生成文件中的类属性一一对应）。

    Args:
        neutral_band: 中性带默认值。
        gate_deadband: GATE 死区默认值。
        has_gate: 公式是否含 GATE 算子。

    Returns:
        参数空间元组（顺序稳定）。
    """
    space: list[ParamSpace] = [
        ParamSpace(
            name="NEUTRAL_BAND",
            default=float(neutral_band),
            kind="float",
            low=0.0,
            high=0.5,
            step=0.005,
            description="中性带：|tanh(factor)| 小于该值视为不交易",
        ),
    ]
    if has_gate:
        space.append(
            ParamSpace(
                name="GATE_DEADBAND",
                default=float(gate_deadband),
                kind="float",
                low=0.0,
                high=0.5,
                step=0.01,
                description=(
                    "GATE 死区（R1）：|condition| 小于该值时沿用上一根选择，"
                    "0 表示关闭保护（与回测逐位一致）"
                ),
            )
        )
    return tuple(space)


def _class_name(spec_name: str) -> str:
    """把策略名转成驼峰类名（``h1_miaosuan_factor`` → ``H1MiaosuanFactorStrategy``）。"""
    cleaned = "".join(ch if ch.isalnum() else " " for ch in spec_name)
    parts = [p for p in cleaned.split() if p]
    if not parts:
        return "MiaosuanFactorStrategy"
    camel = "".join(p[:1].upper() + p[1:] for p in parts)
    if not camel.endswith("Strategy"):
        camel += "Strategy"
    return camel


def _cost_text(cost_sensitivity: Mapping[str, float]) -> str:
    """把成本敏感度字典渲染成一行文本。"""
    if not cost_sensitivity:
        return "n/a"
    items = sorted(cost_sensitivity.items(), key=lambda kv: float(kv[0]))
    return ", ".join(f"{k}x={v:.4f}" for k, v in items)


class ShenjiPort(TargetPort):
    """神机 平台导出端口。

    Args:
        ledger_path: magic 账本路径；``None`` 用默认路径。
        magic: 强制指定 magic（提供则**不访问账本**，便于测试与离线渲染）。
            接受 ``int`` 或数字字符串，渲染前统一归一为 ``int``。
        date: 文件名日期 ``YYYYMMDD``；留空取当天 UTC。
        template_name: 模板文件名。
        gate_deadband: GATE 死区默认值（R1 保护开关，0 = 关闭）。
    """

    platform = PLATFORM_SPEC

    def __init__(
        self,
        *,
        ledger_path: str | Path | None = None,
        magic: int | str | None = None,
        date: str = "",
        template_name: str = DEFAULT_TEMPLATE,
        gate_deadband: float = 0.0,
    ) -> None:
        """初始化端口（不触碰文件系统，除非需要分配 magic）。"""
        self.ledger_path = ledger_path
        self.magic = magic
        self.date = date
        self.template_name = template_name
        self.gate_deadband = float(gate_deadband)

    # ── 四件套 ─────────────────────────────────────────────────────────────
    def extract(self, spec: StrategySpec) -> dict[str, Any]:
        """提取渲染上下文（内核源码、magic、文件名、参数空间等）。"""
        if not isinstance(spec.payload, FactorPayload):
            raise ValueError(
                "神机 因子模板仅支持 FactorPayload；"
                f"当前 payload 为 {type(spec.payload).__name__}（模板参数导出属于模式 B，T04 未实现）"
            )
        tokens = tuple(int(t) for t in spec.payload.tokens)
        if not tokens:
            raise ValueError("FactorPayload.tokens 为空，无法导出")

        plan = build_kernel_plan(tokens)
        op_names = set(plan.op_names)
        has_gate = GATE_TOKEN_NAME in op_names

        gate_token = -1
        gate_func_name = ""
        for tid, oname, _arity, fname in plan.op_entries:
            if oname == GATE_TOKEN_NAME and gate_token < 0:
                gate_token = int(tid)
                gate_func_name = fname

        # 词表版本必须与 payload 一致：token 只有搭配同一词表才有意义。
        if (
            spec.provenance.vocab_version
            and spec.provenance.vocab_version != spec.payload.vocab_version
        ):
            raise ValueError(
                "词表版本不一致：provenance="
                f"{spec.provenance.vocab_version} vs payload={spec.payload.vocab_version}"
            )

        # 硬约束 H-1：词表版本冻结锁。上面的 provenance↔payload 比对可被「同文件一起
        # 篡改」绕过（两者都在 spec 内），因此必须再比对**代码里派生的当前版本**。
        # 版本不匹配（含篡改 / 旧 checkpoint）一律拒绝导出，防止 token 在错误词表下
        # 被解读成别的算子而静默产出错误策略。
        if spec.payload.vocab_version:
            FORMULA_VOCAB.verify(spec.payload.vocab_version)

        # magic 必须是 int（神机 侧 magic: int）；强制指定的可能是字符串（测试 / CLI），
        # 这里统一归一，保证模板渲染出的是裸整数字面量而不是 "661801"。
        raw_magic = self.magic or allocate_magic(
            spec.name or "unnamed", 1, ledger_path=self.ledger_path
        )
        magic = int(raw_magic)
        version = 1
        param_space = build_param_space(
            neutral_band=spec.semantics.neutral_band,
            gate_deadband=self.gate_deadband,
            has_gate=has_gate,
        )
        formula = readable_formula(tokens)

        # 去重：同一 token 可能出现多次，包装函数与字典只需一份。
        seen_features: set[int] = set()
        feature_entries: list[tuple[int, str, str]] = []
        for tid, tname, fname in plan.feature_entries:
            if tid in seen_features:
                continue
            seen_features.add(tid)
            feature_entries.append((tid, tname, fname))
        seen_ops: set[int] = set()
        op_entries: list[tuple[int, str, int, str]] = []
        for tid, oname, arity, fname in plan.op_entries:
            if tid in seen_ops:
                continue
            seen_ops.add(tid)
            op_entries.append((tid, oname, arity, fname))

        return {
            "filename": default_filename(spec.name or "unnamed", version, date=self.date),
            "magic": magic,
            "version": version,
            "vocab_version": spec.payload.vocab_version,
            "tokens": tokens,
            "formula_text": formula,
            "plan": plan,
            "feature_entries": feature_entries,
            "op_entries": op_entries,
            "feat_offset": FORMULA_VOCAB.operator_offset,
            "has_gate": has_gate,
            "gate_token": gate_token,
            "gate_func_name": gate_func_name,
            "param_space": param_space,
            "class_name": _class_name(spec.name),
        }

    def render(self, spec: StrategySpec, ctx: Mapping[str, Any]) -> str:
        """渲染 神机 策略源码。"""
        plan: KernelPlan = ctx["plan"]
        param_space: tuple[ParamSpace, ...] = ctx["param_space"]
        evidence = spec.evidence
        changelog = self._changelog(spec, ctx)

        return self._environment().get_template(self.template_name).render(
            name=spec.name or "unnamed",
            class_name=ctx["class_name"],
            magic=ctx["magic"],
            version=ctx["version"],
            # VOCAB_VERSION 常量取 payload 版本（extract 已校验非空且与内核一致）；
            # provenance.vocab_version 仅作溯源展示，可能为空，不用于冻结锁。
            vocab_version=ctx["vocab_version"],
            changelog=changelog,
            deployable=bool(evidence.deployable),
            research_only=not bool(evidence.deployable),
            module_header=list(self.platform.module_header),
            base_classes=list(self.platform.base_classes),
            symbol=self.platform.default_symbol,
            kernel_blocks=list(plan.blocks),
            tokens=list(ctx["tokens"]),
            n_tokens=len(ctx["tokens"]),
            formula_text=ctx["formula_text"],
            feature_entries=list(ctx["feature_entries"]),
            op_entries=list(ctx["op_entries"]),
            feat_offset=ctx["feat_offset"],
            has_gate=bool(ctx["has_gate"]),
            gate_token=ctx["gate_token"],
            gate_func_name=ctx["gate_func_name"],
            gate_deadband=self.gate_deadband,
            roll_window=int(spec.semantics.roll_window),
            warmup_bars=int(spec.semantics.warmup_bars),
            neutral_band=float(spec.semantics.neutral_band),
            long_short=bool(spec.semantics.long_short),
            # 风控默认（渲染成可调类属性；见本模块顶部常量）。
            sl_atr_mult=SL_ATR_MULT,
            tp_atr_mult=TP_ATR_MULT,
            min_sl_points=MIN_SL_POINTS,
            min_tp_points=MIN_TP_POINTS,
            atr_period=ATR_PERIOD,
            param_space=[p.to_dict() for p in param_space],
            evidence={
                "val_score": evidence.val_score,
                "wf_folds": evidence.wf_folds,
                "deflated_sharpe": evidence.deflated_sharpe,
                "n_trials": evidence.n_trials,
                "gate_verdict": evidence.gate_verdict,
                "gate_reasons": list(evidence.gate_reasons),
                "cost_sensitivity_text": _cost_text(evidence.cost_sensitivity),
                "holdout_sharpe_text": (
                    f"{evidence.holdout_sharpe:.4f}"
                    if evidence.holdout_sharpe is not None
                    else "未消费 hold-out"
                ),
            },
            provenance={
                "git_sha": spec.provenance.git_sha,
                "vocab_version": spec.provenance.vocab_version,
                "data_fingerprint": spec.provenance.data_fingerprint,
                "seed": spec.provenance.seed,
                "market": spec.provenance.market,
                "budget": spec.provenance.budget,
            },
            spec_id=spec.spec_id,
            generated_at=spec.provenance.created_at or "",
        )

    def lint(self, source: str, spec: StrategySpec | None = None) -> list[LintIssue]:
        """对渲染结果做静态检查（含 repaint 检查）。"""
        return lint_source(source)

    # ── 便捷入口 ───────────────────────────────────────────────────────────
    def export(self, spec: StrategySpec) -> ExportResult:
        """等价于 :meth:`TargetPort.compile`，语义更明确的别名。"""
        return self.compile(spec)

    def _changelog(self, spec: StrategySpec, ctx: Mapping[str, Any]) -> list[str]:
        """构造 ``STRATEGY_CHANGELOG`` 条目（按时间正序，最后一条为当前版本）。"""
        evidence = spec.evidence
        return [
            f"v{ctx['version']} 由妙算自动生成：{ctx['formula_text']}",
            f"v{ctx['version']} IR spec_id={spec.spec_id} "
            f"vocab={spec.provenance.vocab_version or ctx['vocab_version']}",
            f"v{ctx['version']} 证据：val_score={evidence.val_score} "
            f"wf_folds={evidence.wf_folds} n_trials={evidence.n_trials} "
            f"dsr={evidence.deflated_sharpe} gate={evidence.gate_verdict}",
            f"v{ctx['version']} 溯源：git={spec.provenance.git_sha} "
            f"data={spec.provenance.data_fingerprint[:12] or 'n/a'} "
            f"seed={spec.provenance.seed}",
        ]

    def _environment(self) -> Environment:
        """构造 Jinja2 环境（未定义变量直接报错，避免静默渲染出坏代码）。"""
        env = Environment(
            loader=FileSystemLoader(str(_TEMPLATE_DIR)),
            undefined=StrictUndefined,
            keep_trailing_newline=True,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        # 中文注释/说明直接落盘，不做 \uXXXX 转义（生成文件要给人读）。
        env.policies["json.dumps_kwargs"] = {"ensure_ascii": False, "sort_keys": True}
        return env
