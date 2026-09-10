"""报告产物层（IO 半边，M5）。

架构把「纯计算」放 ``core/``（无 IO），把「产物持久化」移出。AM ``model_core/evaluator.py``
的 ``save_report`` / ``load_report`` 直接做文件读写（``open`` / ``os.replace`` / ``os.makedirs``），
违反妙算「``core/`` 无文件 IO」铁律，故拆到本模块：

* :func:`report_to_dict` / :func:`report_from_dict` —— 纯序列化（复用 ``core.evaluator`` 的
  ``row_to_dict`` / ``dict_to_row``）；
* :func:`save_report` / :func:`load_report` —— 原子写读（临时文件 + ``os.replace``）。

T02 的 ``report/metrics.py`` 会在此基础上扩展可视化/指标；本模块只保留 AM 语义的最小 IO。
"""

from __future__ import annotations

import json
import os
from typing import Any

from ..core.evaluator import (
    Report,
    ReportRow,
    dict_to_row,
    row_to_dict,
)

__all__ = [
    "ReportPersistError",
    "load_report",
    "report_from_dict",
    "report_to_dict",
    "save_report",
]


class ReportPersistError(Exception):
    """报告持久化失败，保留旧文件并抛出（AM R7.6）。"""


def report_to_dict(report: Report) -> dict[str, Any]:
    """将 :class:`Report` 序列化为 JSON-ready dict。"""
    return {
        "vocab_version": report.vocab_version,
        "generated_at": report.generated_at,
        "config": report.config,
        "rows": [row_to_dict(r) for r in report.rows],
        "active_subset": report.active_subset,
    }


def report_from_dict(data: dict[str, Any]) -> Report:
    """从 JSON dict 反序列化 :class:`Report`。"""
    rows: list[ReportRow] = [dict_to_row(r) for r in data.get("rows", [])]
    return Report(
        vocab_version=data.get("vocab_version", ""),
        generated_at=data.get("generated_at", ""),
        config=data.get("config", {}),
        rows=rows,
        active_subset=data.get("active_subset", []),
    )


def save_report(report: Report, path: str) -> None:
    """持久化报告到 JSON 文件（R7.5/7.6）。

    先写临时文件再原子替换，失败时保留旧文件并抛 :class:`ReportPersistError`。
    """
    tmp_path = path + ".tmp"
    data = report_to_dict(report)
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    except Exception as e:  # noqa: BLE001 - 统一转 ReportPersistError
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:  # noqa: BLE001 - 清理失败不影响主错误
            pass
        raise ReportPersistError(f"报告持久化失败，旧文件已保留: {e}") from e


def load_report(path: str) -> Report:
    """从 JSON 文件加载报告（R7.5）。"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return report_from_dict(data)
