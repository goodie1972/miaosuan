# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""回归验证（regression）测试的共享装置 —— 加载冻结基准数据。

架构 §6.3：妙算的回归验证基于冻结快照——词表版本字符串、特征名序列、算子名序列
均由确定性 sha256 派生，冻结后作为离线基准。每次 CI 运行时与当前实现逐字符比对，
确保核心实现不被意外修改。

冻结基准的两级策略
------------------

1. **冻结快照（始终可用）**：``tests/fixtures/am_vocab_snapshot.json`` 记录了词表的
   版本字符串与有序 token 名称。作为稳定、离线、零外部依赖的回归验证基准。

2. **数值基准（可选）**：``tests/fixtures/*.npz`` 存放了特征/算子/VM/回测的数值快照，
   由 ``scripts/gen_*_baseline.py`` 生成。有则比对，无则跳过。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# ── 路径 ───────────────────────────────────────────────────────────────────

#: 冻结快照
_FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "am_vocab_snapshot.json"

#: 数值基准目录
_FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
_M5_NPZ = _FIXTURES_DIR / "m5_baseline.npz"
_M5_JSON = _FIXTURES_DIR / "m5_baseline.json"
_E2E_NPZ = _FIXTURES_DIR / "e2e_xauusd_baseline.npz"
_E2E_JSON = _FIXTURES_DIR / "e2e_xauusd_meta.json"


# ── 装置 ───────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def frozen_snapshot() -> dict[str, Any]:
    """冻结的词表快照（离线基准）。"""
    with _FIXTURE.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# ── 数值基准（vm / signal / backtest / evaluator）────────────────────────


@pytest.fixture(scope="session")
def m5_npz() -> dict[str, np.ndarray]:
    """M5 数值基准。"""
    if not _M5_NPZ.is_file():
        pytest.skip(f"缺少 M5 基准：{_M5_NPZ}（先跑 scripts/gen_m5_baseline.py）")
    with np.load(_M5_NPZ) as data:
        return {k: data[k] for k in data.files}


@pytest.fixture(scope="session")
def m5_meta() -> dict[str, Any]:
    """M5 元信息与标量结果。"""
    if not _M5_JSON.is_file():
        pytest.skip(f"缺少 M5 元信息：{_M5_JSON}")
    return json.loads(_M5_JSON.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def e2e_npz() -> dict[str, np.ndarray]:
    """端到端 XAUUSD 基准（原始面板 + 特征 + 因子）。"""
    if not _E2E_NPZ.is_file():
        pytest.skip(f"缺少端到端基准：{_E2E_NPZ}（先跑 scripts/gen_e2e_xauusd_baseline.py）")
    with np.load(_E2E_NPZ) as data:
        return {k: data[k] for k in data.files}


@pytest.fixture(scope="session")
def e2e_meta() -> dict[str, Any]:
    """端到端元信息（形状 / 公式 / 统计量）。"""
    if not _E2E_JSON.is_file():
        pytest.skip(f"缺少端到端元信息：{_E2E_JSON}")
    return json.loads(_E2E_JSON.read_text(encoding="utf-8"))
