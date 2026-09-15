# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""目标平台适配层（M16.5）。

本包是**平台无关层**：只依赖 :mod:`miaosuan.ir`，不依赖 CLI、不依赖 ``tune``。
每个具体平台（如 :mod:`miaosuan.adapters.shenji`）提供一个 :class:`TargetPort`
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
