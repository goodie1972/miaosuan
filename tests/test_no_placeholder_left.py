# -*- coding: utf-8 -*-
# Copyright (c) 2026 Original Author
# Copyright (c) 2026 Your Company
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""占位符哨兵：``core/`` 内**不得**残留任何未实现的占位（M3/M4 验收后）。

以文本扫描 ``src/miaosuan/core/*.py``，禁止出现：

* ``NotImplementedError``（占位 compute 的显式失败标志）；
* ``_pending_feature`` / ``_pending_operator``（脚手架占位函数命名）；
* ``TODO`` / ``FIXME`` / ``XXX``（未完成标记）。

该哨兵在 M2（vocab 脚手架）时会失败——这正是它的用途：等 ops/features 全部 numpy 化
落地后转绿，防止后续再把占位实现混入核心域。
"""

from __future__ import annotations

from pathlib import Path

import pytest

CORE_DIR = Path(__file__).resolve().parent.parent / "src" / "miaosuan" / "core"

_FORBIDDEN_MARKERS: tuple[str, ...] = (
    "NotImplementedError",
    "_pending_feature",
    "_pending_operator",
    "TODO",
    "FIXME",
    "XXX",
)


def _core_files() -> list[Path]:
    files = sorted(CORE_DIR.glob("*.py"))
    assert files, f"未找到 core 源文件：{CORE_DIR}"
    return files


@pytest.mark.parametrize("path", _core_files(), ids=lambda p: p.name)
def test_no_placeholder_left(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    hits = [m for m in _FORBIDDEN_MARKERS if m in source]
    assert not hits, f"{path.name} 残留占位/未完成标记：{hits}"


def test_core_has_expected_modules() -> None:
    names = {p.name for p in _core_files()}
    assert {"registry.py", "features.py", "ops.py", "vocab.py"} <= names
