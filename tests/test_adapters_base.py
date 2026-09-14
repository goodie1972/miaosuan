# -*- coding: utf-8 -*-
# Copyright (c) 2026 Original Author
# Copyright (c) 2026 Your Company
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""M16.5：平台无关层（ParamSpace / PlatformSpec / TargetPort / 占位模块）。"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from typing import Any

import pytest

from miaosuan.adapters.base import (
    ExportResult,
    LintIssue,
    LintSeverity,
    ParamSpace,
    PlatformSpec,
    TargetPort,
    install_stub_modules,
    uninstall_stub_modules,
)
from miaosuan.ir.schema import FactorPayload, StrategySpec


def _spec() -> StrategySpec:
    return StrategySpec(
        name="demo",
        payload=FactorPayload(tokens=(0, 1, 65), vocab_version="v9217a2c0d91a"),
    )


def test_param_space_roundtrip() -> None:
    original = ParamSpace(
        name="NEUTRAL_BAND",
        default=0.05,
        kind="float",
        low=0.0,
        high=0.5,
        step=0.005,
        description="中性带",
    )
    restored = ParamSpace.from_dict(original.to_dict())
    assert restored == original


def test_param_space_choices_default_empty() -> None:
    space = ParamSpace.from_dict({"name": "MODE", "default": 1.0, "kind": "choice"})
    assert space.choices == ()
    assert space.low is None and space.high is None


def test_platform_spec_is_data_driven() -> None:
    spec = PlatformSpec(
        name="demo",
        stub_modules={"a.b": ("Thing",)},
        base_classes=("Base",),
        module_header=("import numpy as np",),
    )
    as_dict = spec.to_dict()
    assert as_dict["stub_modules"] == {"a.b": ["Thing"]}
    assert as_dict["base_classes"] == ["Base"]


class _DummyPort(TargetPort):
    platform = PlatformSpec(name="dummy")

    def extract(self, spec: StrategySpec) -> dict[str, Any]:
        return {
            "filename": f"{spec.name}.py",
            # 故意给字符串：验证平台无关层会归一成 int（老调用方可能仍在传字符串）
            "magic": "661801",
            "param_space": (ParamSpace(name="P", default=1.0),),
        }

    def render(self, spec: StrategySpec, ctx: Mapping[str, Any]) -> str:
        return f"# {spec.name}
"

    def lint(self, source: str, spec: StrategySpec | None = None) -> list[LintIssue]:
        return [LintIssue("D001", LintSeverity.WARNING, "warn", 1)]


def test_target_port_compile_orchestrates() -> None:
    result = _DummyPort().compile(_spec())
    assert isinstance(result, ExportResult)
    assert result.ok
    assert result.filename == "demo.py"
    assert result.magic == 661801
    assert isinstance(result.magic, int)
    assert result.param_space[0].name == "P"
    assert len(result.warnings) == 1
    assert result.errors == ()


class _FailingPort(_DummyPort):
    def lint(self, source: str, spec: StrategySpec | None = None) -> list[LintIssue]:
        return [LintIssue("D002", LintSeverity.ERROR, "boom", 2)]


def test_export_result_ok_is_false_when_errors() -> None:
    result = _FailingPort().compile(_spec())
    assert not result.ok
    assert len(result.errors) == 1


def test_target_port_is_abstract() -> None:
    with pytest.raises(TypeError):
        TargetPort()  # type: ignore[abstract]


def test_install_and_uninstall_stub_modules() -> None:
    spec = PlatformSpec(name="demo", stub_modules={"zz_pkg.mod": ("Widget",)})
    created = install_stub_modules(spec)
    try:
        assert "zz_pkg" in created and "zz_pkg.mod" in created
        assert "zz_pkg" in sys.modules and "zz_pkg.mod" in sys.modules
        assert hasattr(sys.modules["zz_pkg.mod"], "Widget")
    finally:
        uninstall_stub_modules(created)
    assert "zz_pkg" not in sys.modules
    assert "zz_pkg.mod" not in sys.modules


def test_install_stub_modules_does_not_clobber_real_modules() -> None:
    spec = PlatformSpec(name="demo", stub_modules={"json": ("dumps",)})
    created = install_stub_modules(spec)
    assert created == ()
    assert sys.modules["json"].__name__ == "json"


def test_lint_issue_to_dict_is_stable() -> None:
    issue = LintIssue("AF001", LintSeverity.ERROR, "repaint", 12)
    assert issue.to_dict() == {
        "code": "AF001",
        "severity": "ERROR",
        "message": "repaint",
        "line": 12,
    }
