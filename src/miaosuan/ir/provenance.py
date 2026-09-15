# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""可追溯信封的构造与采集（M13）。

:class:`~miaosuan.ir.schema.Provenance` 回答四个问题：

* **哪个代码** —— ``git_sha``；
* **哪套词表** —— ``vocab_version``（token 只有在同一词表下才有意义）；
* **哪份数据** —— ``data_fingerprint``（``Panel.fingerprint``）；
* **哪个随机性** —— ``seed`` + ``config_snapshot``。

关于 IO：本模块是 IR 的一部分，原则上不做 IO，但「读取 git sha」天然需要
``subprocess``。为避免在 core / search 层散落 IO，这里集中提供
:func:`resolve_git_sha`，并规定**只有 CLI 边界**可以调用它（其他层应把 sha
作为参数传入）。仓库外 / 无 git 时返回 ``"unknown"``，永不抛异常。
"""

from __future__ import annotations

import subprocess  # noqa: S404 - 仅用于读取本机 git 元信息，无用户输入
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .schema import Provenance

__all__ = [
    "MIAOSUAN_VERSION",
    "build_provenance",
    "resolve_git_sha",
    "short_sha",
    "utc_now_iso",
]

#: 妙算版本号（与 pyproject 保持一致；此处硬编码以避免运行期读取文件）。
MIAOSUAN_VERSION: str = "0.1.0"


def utc_now_iso() -> str:
    """返回当前 UTC 时刻的 ISO 8601 字符串（秒精度，``+00:00`` 后缀）。"""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def short_sha(sha: str, length: int = 12) -> str:
    """截断 git sha 为短形式。

    Args:
        sha: 完整 sha 或 ``unknown``。
        length: 保留长度。

    Returns:
        短 sha；输入为 ``unknown`` / 空串时原样返回。
    """
    if not sha or sha == "unknown":
        return sha
    return sha[:length]


def resolve_git_sha(cwd: str | Path | None = None) -> str:
    """读取当前仓库 HEAD 的 commit sha（**仅 CLI 可调用**）。

    Args:
        cwd: 仓库路径；``None`` 表示进程当前工作目录。

    Returns:
        完整 sha；非 git 仓库 / git 不可用 / 命令失败时返回 ``"unknown"``。
    """
    try:
        completed = subprocess.run(  # noqa: S603 - 固定参数，无 shell
            ["git", "rev-parse", "HEAD"],
            cwd=None if cwd is None else str(cwd),
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if completed.returncode != 0:
        return "unknown"
    return completed.stdout.strip() or "unknown"


def build_provenance(
    *,
    vocab_version: str,
    data_fingerprint: str = "",
    seed: int = 0,
    market: str = "",
    budget: str = "standard",
    git_sha: str = "unknown",
    created_at: str = "",
    config_snapshot: Mapping[str, Any] | None = None,
) -> Provenance:
    """构造 :class:`Provenance`（纯函数，除默认值外不访问环境）。

    Args:
        vocab_version: 词表版本（必填，通常取 ``VOCAB_VERSION``）。
        data_fingerprint: 数据指纹。
        seed: 随机种子。
        market: 市场/品种 profile 名。
        budget: 预算档位。
        git_sha: 代码 sha；由调用方（CLI）用 :func:`resolve_git_sha` 取到后传入。
        created_at: 生成时刻；留空则取当前 UTC（由调用方显式传入更利于测试确定性）。
        config_snapshot: 生效配置快照。

    Returns:
        溯源信封。
    """
    return Provenance(
        git_sha=git_sha,
        vocab_version=vocab_version,
        data_fingerprint=data_fingerprint,
        seed=int(seed),
        created_at=created_at or utc_now_iso(),
        miaosuan_version=MIAOSUAN_VERSION,
        market=market,
        budget=budget,
        config_snapshot=dict(config_snapshot or {}),
    )
