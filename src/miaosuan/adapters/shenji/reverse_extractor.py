# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""反向提取器：从妙算导出的 .py 文件中提取嵌入的 _MIAOSUAN_SPEC_JSON，
并将其还原为 StrategySpec 对象。

使用方法：
    from miaosuan.adapters.shenji.reverse_extractor import extract_spec_from_py
    spec = extract_spec_from_py("path/to/strategy.py")
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

from ...ir.schema import StrategySpec
from ...errors import MiaoSuanError


_SPEC_VAR_PATTERN = re.compile(
    r"_MIAOSUAN_SPEC_JSON\s*=\s*r?\"\"\"([\s\S]*?)\"\"\"",
    re.MULTILINE,
)


def extract_spec_from_py(file_path: str | Path) -> StrategySpec:
    """从导出的 .py 文件中提取嵌入的完整 StrategySpec JSON。

    参数
    ----
    file_path : str | Path
        要读取的 .py 文件路径。

    返回
    ----
    StrategySpec
        还原的策略规格对象。

    异常
    ----
    MiaoSuanError
        如果文件不存在、不是有效的妙算导出文件或 JSON 解析失败。
    """
    path = Path(file_path)
    if not path.is_file():
        raise MiaoSuanError(f"文件不存在：{path}")

    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise MiaoSuanError(f"无法读取文件 {path}：{exc}") from exc

    match = _SPEC_VAR_PATTERN.search(source)
    if not match:
        raise MiaoSuanError(
            f"文件 {path} 不含有效的 _MIAOSUAN_SPEC_JSON 嵌入，"
            "可能不是妙算导出的策略文件或版本过旧。"
        )

    json_text = match.group(1)
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise MiaoSuanError(f"解析 _MIAOSUAN_SPEC_JSON 失败：{exc}") from exc

    try:
        spec = StrategySpec.from_dict(data)
    except Exception as exc:  # noqa: BLE001
        raise MiaoSuanError(f"从字典重建 StrategySpec 失败：{exc}") from exc

    return spec