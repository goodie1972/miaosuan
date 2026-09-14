"""神机 目标平台适配器（M15 / M16）。

提供：

* :class:`ShenjiPort` —— :class:`~miaosuan.adapters.base.TargetPort` 实现；
* :data:`PLATFORM_SPEC` —— 平台静态描述（含占位模块，便于离线 import）；
* :func:`lint_source` —— 生成文件静态检查（含 repaint 检查）；
* :func:`allocate_magic` —— magic 号段账本分配；
* :class:`ShenjiReadOnlyClient` —— 后端**只读**客户端（03 实时页，绝不下单）。
"""

from .contract import PLATFORM_SPEC
from .generator import ShenjiPort, build_param_space, readable_formula
from .lint import lint_source
from .magic_registry import MagicLedger, allocate_magic
from .realtime import (
    ALLOWED_PATHS,
    ShenjiReadOnlyClient,
    BackendUnreachable,
    ReadOnlyViolation,
    RealtimeOutcome,
    safe_call,
)

__all__ = [
    "ALLOWED_PATHS",
    "PLATFORM_SPEC",
    "ShenjiPort",
    "ShenjiReadOnlyClient",
    "BackendUnreachable",
    "MagicLedger",
    "ReadOnlyViolation",
    "RealtimeOutcome",
    "allocate_magic",
    "build_param_space",
    "lint_source",
    "readable_formula",
    "safe_call",
]
