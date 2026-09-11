"""``StrategySpec`` 的确定性序列化 / 反序列化（M13）。

确定性三条铁律：

1. **键排序**：``sort_keys=True``，任何层级都按字典序落盘；
2. **紧凑分隔**：``separators=(",", ":")``，消除空格差异；
3. **非 ASCII 直通**：``ensure_ascii=False`` + 固定 ``utf-8`` 编码。

于是同一 :class:`StrategySpec` 两次导出得到**逐字节相同**的 JSON，可以直接做
diff / 快照比对 / 内容寻址。

注意：本模块提供 :func:`read_spec` / :func:`write_spec` 两个 IO 便捷函数，
但按架构约束，**只有** :mod:`miaosuan.cli` 可以调用它们（IR 层自身不触发 IO）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import StrategySpec

__all__ = [
    "canonical_json",
    "decode_spec",
    "encode_spec",
    "read_spec",
    "write_spec",
]


def canonical_json(obj: Any) -> str:
    """把任意可 JSON 化对象序列化为**确定性**字符串。

    Args:
        obj: 仅含 dict / list / str / int / float / bool / None 的对象。

    Returns:
        键排序、紧凑分隔、UTF-8 友好的 JSON 文本。
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def encode_spec(spec: StrategySpec, *, indent: int | None = 2) -> str:
    """把 :class:`StrategySpec` 编码为确定性 JSON 文本。

    Args:
        spec: 待编码的策略规格。
        indent: 缩进；``None`` 表示紧凑单行。默认 2（便于人读与 diff）。

    Returns:
        JSON 文本（键排序；``indent`` 不影响确定性，只影响换行）。
    """
    if indent is None:
        return canonical_json(spec.to_dict())
    return json.dumps(
        spec.to_dict(), sort_keys=True, indent=indent, ensure_ascii=False
    )


def decode_spec(text: str) -> StrategySpec:
    """把 JSON 文本解码为 :class:`StrategySpec`。

    Args:
        text: :func:`encode_spec` 的产出（或任意等价 JSON）。

    Returns:
        还原后的策略规格。

    Raises:
        json.JSONDecodeError: 文本不是合法 JSON。
        ValueError: JSON 合法但 payload.kind 未知（由
            :meth:`StrategySpec.from_dict` 抛出）。
        KeyError: 缺少 ``payload`` 等必需字段。
    """
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"spec 顶层必须是 JSON 对象，实际为 {type(data).__name__}")
    return StrategySpec.from_dict(data)


def write_spec(path: str | Path, spec: StrategySpec, *, indent: int | None = 2) -> Path:
    """把 spec 写入文件（**仅 CLI 可调用**）。

    Args:
        path: 目标路径（父目录不存在时自动创建）。
        spec: 待写入的策略规格。
        indent: 缩进，语义同 :func:`encode_spec`。

    Returns:
        实际写入的路径。
    """
    target = Path(path)
    if target.parent and not target.parent.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(encode_spec(spec, indent=indent), encoding="utf-8")
    return target


def read_spec(path: str | Path) -> StrategySpec:
    """从文件读取 spec（**仅 CLI 可调用**）。

    Args:
        path: JSON 文件路径。

    Returns:
        还原后的策略规格。
    """
    return decode_spec(Path(path).read_text(encoding="utf-8"))
