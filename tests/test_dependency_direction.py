# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""依赖方向检查（CI 强制，架构 §1.2 / §9.7）。

以 AST 静态扫描 ``src/miaosuan/core/*.py``，断言核心域满足铁律：

1. **不得** ``import torch``（本项目彻底去 torch）；
2. **不得** import ``adapters`` / ``tune`` / ``cli``（禁止 core 反向依赖上层）；
3. **不得**读取环境变量（``os.environ`` / ``os.getenv`` / ``os.putenv`` …）；
4. **不得**做文件 IO（``open`` / ``Path.read_text`` / ``io.open`` …）。

用 AST 而非字符串匹配，避免把文档字符串/注释里的 "torch" 误判为依赖。
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

CORE_DIR = Path(__file__).resolve().parent.parent / "src" / "miaosuan" / "core"

# 禁止导入的顶层模块（含其子模块）
_FORBIDDEN_TOP_MODULES = {"torch"}

# 禁止 core 依赖的上层包名（绝对或相对）
_FORBIDDEN_LAYER_NAMES = {"adapters", "tune", "cli"}

# 被视为「文件 IO」的函数/方法名
_IO_CALL_NAMES = {"open"}

# 被视为「文件 IO」的路径对象方法
_IO_ATTR_NAMES = {
    "read_text",
    "write_text",
    "read_bytes",
    "write_bytes",
    "mkdir",
    "unlink",
    "rmdir",
    "rename",
    "replace",
    "touch",
    "open",
}

# 环境变量读取：os 的成员
_ENV_ATTR_NAMES = {"environ", "getenv", "putenv"}


def _core_files() -> list[Path]:
    files = sorted(CORE_DIR.glob("*.py"))
    assert files, f"未找到 core 源文件：{CORE_DIR}"
    return files


def _iter_violations(tree: ast.AST) -> list[str]:
    violations: list[str] = []

    for node in ast.walk(tree):
        # ── import 检查 ──────────────────────────────────────────────
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in _FORBIDDEN_TOP_MODULES:
                    violations.append(f"禁止 import {alias.name!r}（core 不得依赖 torch）")

        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            top = module.split(".")[0]
            if node.level == 0:
                if top in _FORBIDDEN_TOP_MODULES:
                    violations.append(f"禁止 from {module!r} import（torch 依赖）")
                # 绝对导入上层包
                parts = module.split(".")
                if (
                    len(parts) >= 2
                    and parts[0] == "miaosuan"
                    and parts[1] in _FORBIDDEN_LAYER_NAMES
                ):
                    violations.append(f"禁止 core 依赖上层包 {module!r}")
            else:
                # 相对导入：解析第一段名称
                if module:
                    first = module.split(".")[0]
                    if first in _FORBIDDEN_LAYER_NAMES:
                        violations.append(f"禁止 core 相对导入上层包 {module!r}")

        # ── 环境变量读取 ─────────────────────────────────────────────
        elif isinstance(node, ast.Attribute):
            if node.attr in _ENV_ATTR_NAMES and _is_os(node.value):
                violations.append(f"禁止读取环境变量：os.{node.attr}")

        # ── 文件 IO：函数调用 ────────────────────────────────────────
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in _IO_CALL_NAMES:
                violations.append(f"禁止文件 IO：调用 {func.id}()")
            elif isinstance(func, ast.Name) and func.id == "Path":
                violations.append("禁止文件 IO：构造 pathlib.Path")
            elif (
                isinstance(func, ast.Attribute)
                and func.attr in _IO_ATTR_NAMES
                and not _is_os(func.value)  # 排除 os.environ.get 之类的误报
            ):
                violations.append(f"禁止文件 IO：调用 .{func.attr}()")

    return violations


def _is_os(node: ast.AST) -> bool:
    return isinstance(node, ast.Name) and node.id == "os"


@pytest.mark.parametrize("path", _core_files(), ids=lambda p: p.name)
def test_core_dependency_direction(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    violations = _iter_violations(tree)
    assert not violations, f"{path.name} 违反依赖方向铁律：\n  - " + "\n  - ".join(violations)


def test_core_files_exist() -> None:
    names = {p.name for p in _core_files()}
    assert {"registry.py", "features.py", "ops.py", "vocab.py"} <= names
