# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""M3 算子对拍的**确定性输入用例**（纯 numpy，无 torch / 无妙算依赖）。

同一份用例被两侧共享：

* ``scripts/gen_ops_baseline.py``（torch 环境）→ 生成 AM 原实现的基准输出；
* ``tests/parity/test_ops_parity.py``（妙算 venv）→ 用妙算 numpy 实现复算并比对。

由于两侧 numpy 版本一致（2.5.3）且 seed 固定，``build_cases()`` 在两侧**逐位相同**；
生成器仍会把输入一并写入基准文件，测试侧再做一次输入一致性校验，防止 RNG 漂移。

用例覆盖（含 team-lead 要求的 0 / 负值 / 极大值 / NaN 附近）：

  ==========  ================================  ==============
  case        含义                               容差 (atol, rtol)
  ==========  ================================  ==============
  base        标准正态，O(1) 量级                 (1e-5, 1e-5)
  mixed       含大量 0 与负值                     (1e-4, 1e-4)
  single      N=1（跨截面算子恒等退化路径）        (1e-4, 1e-4)
  const       含常数列（零跨度 / 退化窗口）        (1e-4, 1e-4)
  extreme     幅度 ±1e4，少数 ±1e6（溢出处）      (1e-3, 1e-4)
  nan         注入 NaN（NaN 传播 / nan_to_num）    (1e-4, 1e-4)
  ==========  ================================  ==============
"""

from __future__ import annotations

from typing import Any

import numpy as np

# 全局随机种子（与 AppConfig 默认 seed 一致）
SEED = 20260910

# dtype：与 AM 生产张量一致（float32）
DTYPE = np.float32


def _case(x: np.ndarray, y: np.ndarray, z: np.ndarray, atol: float, rtol: float) -> dict[str, Any]:
    return {
        "x": x.astype(DTYPE, copy=False),
        "y": y.astype(DTYPE, copy=False),
        "z": z.astype(DTYPE, copy=False),
        "atol": atol,
        "rtol": rtol,
    }


def build_cases() -> dict[str, dict[str, Any]]:
    """构造全部确定性用例，返回 ``{case_name: {"x","y","z","atol","rtol"}}``。"""
    cases: dict[str, dict[str, Any]] = {}

    # ── base：标准正态 O(1) ────────────────────────────────────────────
    rng = np.random.default_rng(SEED)
    cases["base"] = _case(
        rng.standard_normal((8, 64)),
        rng.standard_normal((8, 64)),
        rng.standard_normal((8, 64)),
        atol=1e-5,
        rtol=1e-5,
    )

    # ── mixed：含大量 0 与负值 ────────────────────────────────────────
    rng = np.random.default_rng(SEED + 1)
    x = rng.standard_normal((6, 48)) * 3.0 - 1.0
    y = rng.standard_normal((6, 48)) * 3.0 - 1.0
    z = rng.standard_normal((6, 48)) * 3.0 - 1.0
    x[x < 0.3] = 0.0  # 约 2 成置零
    y.flat[::7] = 0.0
    z.flat[::11] = 0.0
    cases["mixed"] = _case(x, y, z, atol=1e-4, rtol=1e-4)

    # ── single：N=1（跨截面算子恒等退化）─────────────────────────────
    rng = np.random.default_rng(SEED + 2)
    cases["single"] = _case(
        rng.standard_normal((1, 32)),
        rng.standard_normal((1, 32)),
        rng.standard_normal((1, 32)),
        atol=1e-4,
        rtol=1e-4,
    )

    # ── const：含常数列（零跨度 / 退化窗口）───────────────────────────
    rng = np.random.default_rng(SEED + 3)
    x = rng.standard_normal((4, 32))
    y = rng.standard_normal((4, 32))
    z = rng.standard_normal((4, 32))
    x[:, 0:6] = 2.0  # 前 6 列常数 → 零跨度窗口
    y[:, 10:15] = -1.5
    z[:, 20:25] = 0.0
    cases["const"] = _case(x, y, z, atol=1e-4, rtol=1e-4)

    # ── extreme：大幅度（±1e4，少数 ±1e6，压 clamp/溢出分支）───────────
    rng = np.random.default_rng(SEED + 4)
    x = rng.standard_normal((4, 40)) * 1e4
    y = rng.standard_normal((4, 40)) * 1e4
    z = rng.standard_normal((4, 40)) * 1e4
    x[0, 5] = 1e6
    x[1, 7] = -1e6
    y[2, 9] = 1e6
    z[3, 11] = -1e6
    cases["extreme"] = _case(x, y, z, atol=1e-3, rtol=1e-4)

    # ── nan：注入 NaN（NaN 传播 / nan_to_num）─────────────────────────
    rng = np.random.default_rng(SEED + 5)
    x = rng.standard_normal((4, 40))
    y = rng.standard_normal((4, 40))
    z = rng.standard_normal((4, 40))
    x[0, 3] = np.nan
    x[2, 17] = np.nan
    y[1, 8] = np.nan
    z[3, 25] = np.nan
    cases["nan"] = _case(x, y, z, atol=1e-4, rtol=1e-4)

    return cases


#: NaN 用例中**排序语义未定义一致**、故不参与 NaN 位置比对的算子。
#: 这些算子依赖 argmax / argsort 的排序，而 NaN 在 torch 与 numpy 的排位规则
#: 未承诺一致；其余算子的 NaN 处理均为逐元素 nan_to_num 或 NaN 传播，两侧一致。
NAN_CASE_EXEMPT_OPS: frozenset[str] = frozenset(
    {
        "TS_ARG_MAX_5",
        "TS_ARG_MIN_5",
        "CS_RANK",
    }
)


def case_names() -> list[str]:
    return list(build_cases().keys())
