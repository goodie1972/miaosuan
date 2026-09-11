"""AlgoForge 目标平台适配器（M15 / M16）。

提供：

* :class:`AlgoforgePort` —— :class:`~miaosuan.adapters.base.TargetPort` 实现；
* :data:`PLATFORM_SPEC` —— 平台静态描述（含占位模块，便于离线 import）；
* :func:`lint_source` —— 生成文件静态检查（含 repaint 检查）；
* :func:`allocate_magic` —— magic 号段账本分配。
"""

from .contract import PLATFORM_SPEC
from .generator import AlgoforgePort, build_param_space, readable_formula
from .lint import lint_source
from .magic_registry import MagicLedger, allocate_magic

__all__ = [
    "PLATFORM_SPEC",
    "AlgoforgePort",
    "MagicLedger",
    "allocate_magic",
    "build_param_space",
    "lint_source",
    "readable_formula",
]
