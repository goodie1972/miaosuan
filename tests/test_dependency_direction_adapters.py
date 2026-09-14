"""依赖方向（M16.5 扩展）：``adapters`` 与「唯一 IO/env 边界」。

在既有 ``test_dependency_direction.py``（只管 ``core/``）之外补充两条新规：

1. ``adapters`` **不得** import ``cli`` / ``tune``（导出层不得反向依赖入口与调参层）；
2. ``AppConfig.from_env`` 与 ``os.environ`` **只在** ``cli.py`` 被调用 —— 这是
   「CLI 是唯一允许 IO/env 的层」这条铁律的可执行断言。

注：``adapters/shenji/magic_registry.py`` 允许做文件 IO —— magic 账本是
**持久化状态**（号段一旦分配不可复用），不是临时产物。
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src" / "miaosuan"
ADAPTERS_DIR = SRC_DIR / "adapters"

_FORBIDDEN_LAYER_NAMES = {"cli", "tune"}


def _adapter_files() -> list[Path]:
    files = sorted(ADAPTERS_DIR.rglob("*.py"))
    assert files, f"未找到 adapters 源文件：{ADAPTERS_DIR}"
    return files


def _forbidden_imports(tree: ast.AST) -> list[str]:
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        module = node.module or ""
        if node.level == 0:
            parts = module.split(".")
            if len(parts) >= 2 and parts[0] == "miaosuan" and parts[1] in _FORBIDDEN_LAYER_NAMES:
                violations.append(f"禁止绝对导入上层模块 {module!r}")
        else:
            first = module.split(".")[0] if module else ""
            if first in _FORBIDDEN_LAYER_NAMES:
                violations.append(f"禁止相对导入上层模块 {module!r}")
    return violations


@pytest.mark.parametrize("path", _adapter_files(), ids=lambda p: str(p.relative_to(SRC_DIR)))
def test_adapters_do_not_import_cli_or_tune(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations = _forbidden_imports(tree)
    assert not violations, f"{path.name} 违反依赖方向：\n  - " + "\n  - ".join(violations)


def test_adapters_never_import_torch() -> None:
    for path in _adapter_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert not any(a.name.split(".")[0] == "torch" for a in node.names), path
            elif isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".")[0] != "torch", path


def _all_source_files() -> list[Path]:
    return sorted(SRC_DIR.rglob("*.py"))


def test_from_env_only_called_in_cli() -> None:
    offenders: list[str] = []
    for path in _all_source_files():
        source = path.read_text(encoding="utf-8")
        if "from_env" not in source:
            continue
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and node.attr == "from_env"
                and path.name != "cli.py"
            ):
                offenders.append(f"{path.relative_to(SRC_DIR)}:{node.lineno}")
    assert not offenders, "AppConfig.from_env 只允许在 cli.py 调用：" + ", ".join(offenders)


def test_os_environ_only_in_config_and_cli() -> None:
    allowed = {"config.py", "cli.py"}
    offenders: list[str] = []
    for path in _all_source_files():
        if path.name in allowed:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in {"environ", "getenv", "putenv"}:
                offenders.append(f"{path.relative_to(SRC_DIR)}:{node.lineno} os.{node.attr}")
    assert not offenders, "读取环境变量只允许在 config.py / cli.py：" + ", ".join(offenders)


def test_adapters_do_not_import_search_or_gate() -> None:
    """导出层只依赖 ``core``（取内核源码）与 ``ir``，不得依赖挖掘/门禁层。"""
    offenders: list[str] = []
    for path in _adapter_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or ""):
                parts = (node.module or "").split(".")
                if len(parts) >= 2 and parts[0] == "miaosuan" and parts[1] in {"search", "gate"}:
                    offenders.append(f"{path.name}:{node.lineno} {node.module}")
                elif node.level >= 2 and parts[0] in {"search", "gate"}:
                    offenders.append(f"{path.name}:{node.lineno} .{node.module}")
    assert not offenders, "adapters 不得依赖 search / gate：" + ", ".join(offenders)
