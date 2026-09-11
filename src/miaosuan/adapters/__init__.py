"""目标平台适配层（M16.5）。

本包是**平台无关层**：只依赖 :mod:`miaosuan.ir`，不依赖 CLI、不依赖 ``tune``。
每个具体平台（如 :mod:`miaosuan.adapters.algoforge`）提供一个 :class:`TargetPort`
实现与一份 :class:`PlatformSpec` 描述。
"""

from .base import (
    ExportResult,
    LintIssue,
    LintSeverity,
    ParamSpace,
    PlatformSpec,
    TargetPort,
    install_stub_modules,
    uninstall_stub_modules,
)

__all__ = [
    "ExportResult",
    "LintIssue",
    "LintSeverity",
    "ParamSpace",
    "PlatformSpec",
    "TargetPort",
    "install_stub_modules",
    "uninstall_stub_modules",
]
