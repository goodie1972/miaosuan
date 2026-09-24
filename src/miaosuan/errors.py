# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""统一异常体系与错误码（架构 §9.5）。

设计原则：

* 所有妙算异常继承 :class:`MiaoSuanError`，携带稳定的 ``code`` 字符串，
  便于日志聚合、门禁判定与上层就近捕获；
* **禁止**裸 ``except: pass``（AM ``vm.py:250`` 的 ``except Exception: return None``
  会吞掉真实错误）；妙算改为返回 ``None`` 时记录 warning 并计数；
* 本模块**只依赖标准库**，可被任意层（含 ``core/``）安全导入，不引入反向依赖。

错误码命名规则：``E-<DOMAIN>-<REASON>``，全部大写下划线。
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "MiaoSuanError",
    "ConfigError",
    "DataError",
    "DataFingerprintError",
    "HoldoutSealedError",
    "VocabVersionMismatchError",
    "FormulaStructureError",
    "RepaintError",
    "TargetPortError",
    "LintError",
    "GateError",
]

# 允许出现在错误码中的字符（供 __init__ 做轻量校验）
_CODE_PREFIX: str = "E-"


class MiaoSuanError(Exception):
    """妙算异常基类。

    :param message: 人类可读的错误描述。
    :param code:    覆盖类级默认错误码（形如 ``E-DOMAIN-REASON``）。
    :param context: 结构化上下文（写入日志），例如 ``{"symbol": "XAUUSD"}``。
    """

    #: 类级默认错误码，子类应覆盖
    code: str = "E-MIAOSUAN"

    def __init__(
        self,
        message: str = "",
        *,
        code: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message: str = message
        if code is not None:
            self.code = code
        self.context: dict[str, Any] = dict(context) if context else {}

    def __str__(self) -> str:  # pragma: no cover - 纯展示
        if self.context:
            return f"[{self.code}] {self.message} | {self.context}"
        return f"[{self.code}] {self.message}"

    def to_dict(self) -> dict[str, Any]:
        """序列化为日志/产物友好的字典。"""
        return {"code": self.code, "message": self.message, "context": dict(self.context)}


# ── 配置层 ─────────────────────────────────────────────────────────────────


class ConfigError(MiaoSuanError):
    """配置非法或缺失（架构 §9.2）。"""

    code = "E-CONFIG"


# ── 数据层 ─────────────────────────────────────────────────────────────────


class DataError(MiaoSuanError):
    """数据加载 / 契约错误（Panel 形状、缺失值等，架构 §9.4）。"""

    code = "E-DATA"


class DataFingerprintError(MiaoSuanError):
    """数据指纹校验失败。"""

    code = "E-DATA-FINGERPRINT"


# ── 门禁层 ─────────────────────────────────────────────────────────────────


class HoldoutSealedError(MiaoSuanError):
    """hold-out 指纹已被使用过，拒绝二次 peek（架构 §6.1、§9.5）。"""

    code = "E-HOLDOUT-SEALED"


class GateError(MiaoSuanError):
    """门禁判定过程本身出错（非「未通过门禁」这一正常结果）。"""

    code = "E-GATE"


# ── 核心域 ─────────────────────────────────────────────────────────────────


class VocabVersionMismatchError(MiaoSuanError):
    """加载产物版本 ≠ 当前派生 :data:`VOCAB_VERSION`（架构 §9.3、§9.5）。

    由 ``core.vocab.FormulaVocab.verify()`` 在版本不匹配时抛出；调用方应拒绝
    加载且不消费任何 token。旧 checkpoint / ``best_strategy.json`` 需重新训练/重建。
    """

    code = "E-VOCAB-MISMATCH"


class FormulaStructureError(MiaoSuanError):
    """RPN 公式结构非法（栈深度越界、算子 arity 不匹配等）。"""

    code = "E-FORMULA-STRUCTURE"


# ── 适配层 ─────────────────────────────────────────────────────────────────


class RepaintError(MiaoSuanError):
    """防 repaint 静态检查命中 bar0（``candles[-1]``）等违约（架构 §5.1）。"""

    code = "E-REPAINT-BAR0"


class LintError(MiaoSuanError):
    """契约 / repaint lint 失败。"""

    code = "E-LINT"


class TargetPortError(MiaoSuanError):
    """目标平台适配层（妙算）双向编译错误。"""

    code = "E-TARGETPORT"
