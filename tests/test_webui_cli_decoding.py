# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for full license text.

"""``_decode_cli_bytes`` 的编码回归：CLI 子进程输出必须逐行判编码。

背景：中文 Windows 上 PyInstaller 冻结 exe 忽略 ``PYTHONIOENCODING``/``PYTHONUTF8``，
仍然按 GBK 写 stdout；源码运行的子进程则按 UTF-8 写。整段一把梭的解码只要缓冲区
里混进一行异编码字节，严格 UTF-8 就整体失败、所有中文行一起掉进同一个兜底分支，
用户看到的就是「导出弹窗整屏乱码、ASCII 行却完好」。
"""

from __future__ import annotations

from miaosuan.webui.server import _decode_cli_bytes

_ZH_LINES = ["已产出：D:\\x\\README_zh.md", "已导出：D:\\x\\strategy.py", "门禁=BLOCKED  DEPLOYABLE=False"]


def test_pure_utf8_payload_decodes() -> None:
    raw = "\n".join(_ZH_LINES).encode("utf-8")
    assert _decode_cli_bytes(raw) == "\n".join(_ZH_LINES)


def test_pure_gbk_payload_decodes() -> None:
    raw = "\n".join(_ZH_LINES).encode("gbk")
    assert _decode_cli_bytes(raw) == "\n".join(_ZH_LINES)


def test_mixed_buffer_keeps_every_line_readable() -> None:
    """一行 UTF-8 + 一行 GBK：旧实现整体走 GBK，UTF-8 那行会整行乱码。"""
    expected = [
        "已产出：D:\\x\\README_zh.md",  # UTF-8
        "已导出：D:\\x\\strategy.py",  # GBK
        "magic=662101  name=m30_xauusd_miaosuan",  # 纯 ASCII，两种编码同解
    ]
    raw = (
        expected[0].encode("utf-8")
        + b"\n"
        + expected[1].encode("gbk")
        + b"\r\n"
        + expected[2].encode("ascii")
    )
    assert _decode_cli_bytes(raw).splitlines() == expected


def test_empty_bytes_is_empty_string() -> None:
    assert _decode_cli_bytes(b"") == ""
