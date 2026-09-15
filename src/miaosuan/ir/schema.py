# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""``StrategySpec`` v1 —— 挖掘层与导出层之间的唯一契约（M13）。

四段结构（缺一不可）：

==============  ==========================================================
段              内容
==============  ==========================================================
``payload``     因子体：RPN token 序列（:class:`FactorPayload`）或
                模板参数（:class:`ParamPayload`，模式 B 预留）。
``semantics``   语义：仓位函数、中性带、多空开关、预热根数。
``evidence``    证据：WF 折数、hold-out 夏普、成本敏感度、试验次数、DSR、
                门禁结论。
``provenance``  溯源：git sha、词表版本、数据指纹、随机种子、配置快照。
==============  ==========================================================

设计约束：

* **确定性**：:meth:`StrategySpec.to_dict` 产出的是纯字典，序列化由
  :mod:`miaosuan.ir.codec` 以 ``sort_keys=True`` 落盘，保证同一 spec 字节一致；
* **可追溯**：:meth:`StrategySpec.spec_id` 只由 payload + semantics + 词表版本
  决定，与运行时刻无关，因此同一因子重跑得到同一 id；
* **不变性**：全部为 ``frozen`` dataclass，杜绝导出后被就地篡改。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

#: IR 规格版本。破坏性变更必须递增主版本。
SPEC_VERSION: str = "1.0"

#: 中性带默认值（与 :mod:`miaosuan.core.signal` 的 ``MIN_TRADE_EXPOSURE`` 一致）。
DEFAULT_NEUTRAL_BAND: float = 0.05

#: 预热根数默认值 = 滚动窗口(500) - 1，与 :class:`miaosuan.core.vm.StackVM` 一致。
DEFAULT_WARMUP_BARS: int = 499


@dataclass(frozen=True)
class FactorPayload:
    """因子体：RPN token 序列 + 词表版本。

    Attributes:
        tokens: 前缀（RPN）token id 序列，顺序即求值顺序。
        vocab_version: 词表版本指纹（当前固定 ``v9217a2c0d91a``）。token 只有
            搭配同一词表才有意义，故与 ``tokens`` 强绑定。
    """

    kind: ClassVar[str] = "factor"
    tokens: tuple[int, ...]
    vocab_version: str

    def to_dict(self) -> dict[str, Any]:
        """序列化为纯字典。"""
        return {
            "kind": self.kind,
            "tokens": list(self.tokens),
            "vocab_version": self.vocab_version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FactorPayload:
        """从字典还原（:meth:`to_dict` 的逆操作）。"""
        return cls(
            tokens=tuple(int(t) for t in data["tokens"]),
            vocab_version=str(data["vocab_version"]),
        )


@dataclass(frozen=True)
class ParamPayload:
    """模板参数体（模式 B / tune 预留，T04 只落结构不落实现）。

    Attributes:
        template_id: 目标平台模板标识。
        params: ``{参数名: 数值}``，键必须与模板声明的 ``ParamSpace`` 对齐。
    """

    kind: ClassVar[str] = "param"
    template_id: str
    params: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """序列化为纯字典。"""
        return {
            "kind": self.kind,
            "template_id": self.template_id,
            "params": dict(self.params),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ParamPayload:
        """从字典还原（:meth:`to_dict` 的逆操作）。"""
        return cls(
            template_id=str(data["template_id"]),
            params={str(k): float(v) for k, v in (data.get("params") or {}).items()},
        )


@dataclass(frozen=True)
class Semantics:
    """策略语义：从因子到仓位这一段「平台无关」的约定。

    Attributes:
        position_fn: 仓位映射函数名（当前仅 ``tanh``）。
        neutral_band: 中性带：``|position| < neutral_band`` 置 0（不交易）。
        long_short: 是否允许多空双向。
        warmup_bars: 预热根数：不足则不出信号（与滚动归一化窗口绑定）。
        timeframe: 目标周期（如 ``H1``）。
        roll_window: 因子滚动归一化窗口（默认 500，与 ``StackVM`` 一致）。
    """

    position_fn: str = "tanh"
    neutral_band: float = DEFAULT_NEUTRAL_BAND
    long_short: bool = True
    warmup_bars: int = DEFAULT_WARMUP_BARS
    timeframe: str = "H1"
    roll_window: int = 500

    def to_dict(self) -> dict[str, Any]:
        """序列化为纯字典。"""
        return {
            "position_fn": self.position_fn,
            "neutral_band": self.neutral_band,
            "long_short": self.long_short,
            "warmup_bars": self.warmup_bars,
            "timeframe": self.timeframe,
            "roll_window": self.roll_window,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Semantics:
        """从字典还原（:meth:`to_dict` 的逆操作）。"""
        return cls(
            position_fn=str(data.get("position_fn", "tanh")),
            neutral_band=float(data.get("neutral_band", DEFAULT_NEUTRAL_BAND)),
            long_short=bool(data.get("long_short", True)),
            warmup_bars=int(data.get("warmup_bars", DEFAULT_WARMUP_BARS)),
            timeframe=str(data.get("timeframe", "H1")),
            roll_window=int(data.get("roll_window", 500)),
        )


@dataclass(frozen=True)
class Evidence:
    """搜索/门禁产生的证据（只读快照，绝不因导出而改变）。

    Attributes:
        n_trials: 本次搜索实际评估的公式条数（用于多重检验校正）。
        wf_folds: Walk-Forward 折数。
        val_score: 开发集（验证段）最终得分。
        holdout_sharpe: hold-out 段夏普（未消费封印时为 ``None``）。
        cost_sensitivity: ``{成本倍数: 夏普}``，如 ``{"0.5": 1.2, "2.0": 0.8}``。
        deflated_sharpe: 去膨胀夏普（DSR 的概率值，``[0, 1]``）。
        gate_verdict: 门禁结论（``DEPLOYABLE`` / ``RESEARCH_ONLY`` / ``BLOCKED``）。
        gate_reasons: 门禁触发原因列表。
    """

    n_trials: int = 0
    wf_folds: int = 0
    val_score: float | None = None
    holdout_sharpe: float | None = None
    cost_sensitivity: dict[str, float] = field(default_factory=dict)
    deflated_sharpe: float | None = None
    gate_verdict: str = "UNKNOWN"
    gate_reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """序列化为纯字典。"""
        return {
            "n_trials": self.n_trials,
            "wf_folds": self.wf_folds,
            "val_score": self.val_score,
            "holdout_sharpe": self.holdout_sharpe,
            "cost_sensitivity": dict(self.cost_sensitivity),
            "deflated_sharpe": self.deflated_sharpe,
            "gate_verdict": self.gate_verdict,
            "gate_reasons": list(self.gate_reasons),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Evidence:
        """从字典还原（:meth:`to_dict` 的逆操作）。"""
        return cls(
            n_trials=int(data.get("n_trials", 0)),
            wf_folds=int(data.get("wf_folds", 0)),
            val_score=None if data.get("val_score") is None else float(data["val_score"]),
            holdout_sharpe=(
                None if data.get("holdout_sharpe") is None else float(data["holdout_sharpe"])
            ),
            cost_sensitivity={
                str(k): float(v) for k, v in (data.get("cost_sensitivity") or {}).items()
            },
            deflated_sharpe=(
                None if data.get("deflated_sharpe") is None else float(data["deflated_sharpe"])
            ),
            gate_verdict=str(data.get("gate_verdict", "UNKNOWN")),
            gate_reasons=tuple(str(r) for r in (data.get("gate_reasons") or ())),
        )

    @property
    def deployable(self) -> bool:
        """门禁是否允许上实盘（``DEPLOYABLE``）。"""
        return self.gate_verdict == "DEPLOYABLE"


@dataclass(frozen=True)
class Provenance:
    """可追溯信封：回答「这个策略是哪个代码 + 哪份数据 + 哪个种子跑出来的」。

    Attributes:
        git_sha: 生成时刻的代码 commit sha（无法获取时为 ``unknown``）。
        vocab_version: 词表版本（必须与 payload 中的一致）。
        data_fingerprint: 数据指纹（``Panel.fingerprint``，sha256）。
        seed: 随机种子。
        created_at: 生成时刻（ISO 8601，UTC）。不参与 :meth:`StrategySpec.spec_id`。
        miaosuan_version: 妙算版本号。
        market: 市场/品种 profile 名。
        budget: 预算档位（``quick`` / ``standard`` / ``deep``）。
        config_snapshot: 生效配置快照（CLI 注入，键排序后落盘）。
    """

    git_sha: str = "unknown"
    vocab_version: str = ""
    data_fingerprint: str = ""
    seed: int = 0
    created_at: str = ""
    miaosuan_version: str = "0.1.0"
    market: str = ""
    budget: str = "standard"
    config_snapshot: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """序列化为纯字典。"""
        return {
            "git_sha": self.git_sha,
            "vocab_version": self.vocab_version,
            "data_fingerprint": self.data_fingerprint,
            "seed": self.seed,
            "created_at": self.created_at,
            "miaosuan_version": self.miaosuan_version,
            "market": self.market,
            "budget": self.budget,
            "config_snapshot": dict(self.config_snapshot),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Provenance:
        """从字典还原（:meth:`to_dict` 的逆操作）。"""
        return cls(
            git_sha=str(data.get("git_sha", "unknown")),
            vocab_version=str(data.get("vocab_version", "")),
            data_fingerprint=str(data.get("data_fingerprint", "")),
            seed=int(data.get("seed", 0)),
            created_at=str(data.get("created_at", "")),
            miaosuan_version=str(data.get("miaosuan_version", "0.1.0")),
            market=str(data.get("market", "")),
            budget=str(data.get("budget", "standard")),
            config_snapshot=dict(data.get("config_snapshot") or {}),
        )


@dataclass(frozen=True)
class StrategySpec:
    """策略规格 v1：挖掘层 → 导出层的唯一载体。

    Attributes:
        spec_version: IR 版本（当前 ``1.0``）。
        name: 策略名（用于文件命名与 ``STRATEGY_NAME``）。
        payload: 因子体或模板参数体。
        semantics: 语义段。
        evidence: 证据段。
        provenance: 溯源段。
        notes: 自由文本备注（不参与指纹）。
    """

    spec_version: str = SPEC_VERSION
    name: str = ""
    payload: FactorPayload | ParamPayload = field(
        default_factory=lambda: FactorPayload(tokens=(), vocab_version="")
    )
    semantics: Semantics = field(default_factory=Semantics)
    evidence: Evidence = field(default_factory=Evidence)
    provenance: Provenance = field(default_factory=Provenance)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """序列化为纯字典（键顺序固定，:mod:`codec` 落盘时再按字典序排序）。"""
        return {
            "spec_version": self.spec_version,
            "name": self.name,
            "payload": self.payload.to_dict(),
            "semantics": self.semantics.to_dict(),
            "evidence": self.evidence.to_dict(),
            "provenance": self.provenance.to_dict(),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> StrategySpec:
        """从字典还原（:meth:`to_dict` 的逆操作）。"""
        raw_payload = data.get("payload") or {}
        kind = str(raw_payload.get("kind", "factor"))
        if kind == FactorPayload.kind:
            payload: FactorPayload | ParamPayload = FactorPayload.from_dict(raw_payload)
        elif kind == ParamPayload.kind:
            payload = ParamPayload.from_dict(raw_payload)
        else:
            raise ValueError(f"未知 payload.kind: {kind!r}（期望 'factor' 或 'param'）")
        return cls(
            spec_version=str(data.get("spec_version", SPEC_VERSION)),
            name=str(data.get("name", "")),
            payload=payload,
            semantics=Semantics.from_dict(data.get("semantics") or {}),
            evidence=Evidence.from_dict(data.get("evidence") or {}),
            provenance=Provenance.from_dict(data.get("provenance") or {}),
            notes=str(data.get("notes", "")),
        )

    @property
    def spec_id(self) -> str:
        """内容寻址 id（sha256 前 16 位）。

        只由 ``payload`` + ``semantics`` + 词表版本决定，**不含**时间戳、
        git sha、证据等易变量，因此同一因子重复导出得到同一 id。
        """
        core = {
            "payload": self.payload.to_dict(),
            "semantics": self.semantics.to_dict(),
            "spec_version": self.spec_version,
        }
        return _stable_hash(core)

    @property
    def tokens(self) -> tuple[int, ...]:
        """若为因子体，返回 token 序列；否则返回空元组。"""
        if isinstance(self.payload, FactorPayload):
            return self.payload.tokens
        return ()


def _stable_hash(obj: Any) -> str:
    """对任意可 JSON 化对象做稳定哈希（键排序，避免字典序影响结果）。

    Args:
        obj: 仅含 dict/list/str/int/float/bool/None 的对象。

    Returns:
        sha256 十六进制摘要的前 16 位。
    """
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
