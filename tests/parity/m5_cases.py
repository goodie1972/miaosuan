"""M5 对拍的**确定性输入用例**（纯 numpy，无 torch / 无妙算依赖）。

同一份用例被两侧共享：

* ``scripts/gen_m5_baseline.py``（torch Oracle）→ 生成 AM 原实现的输入/输出基准；
* ``tests/parity/test_{vm,signal,backtest,evaluator}_parity.py``（妙算 venv）→ 用妙算 numpy
  实现复算并比对。

由于两侧 numpy 版本一致（2.5.3）且 seed 固定，``build_*`` 在两侧**逐位相同**；生成器会把
输入一并写入基准文件，测试侧再做一次输入一致性校验，防止 RNG 漂移。

覆盖（对应 team-lead 的 M5 难点与验收）：

* **VM**：合法公式 / 参数不足 / 栈深非法（残留≠1 / 空）/ 特征越界 / 未知算子 / 截断特征张量
  / NaN-Inf（``nan_to_num``）；以及 ``_normalize_output`` 的滚动 500（warm-up 499）/ expanding
  / 截面 / 常数短路。
* **signal**：连续仓位（tanh + 中性带）的 N=1 / N>1 / 近阈值。
* **backtest**：多目标 Reward 的单品种 / 多品种 / OOS 门控。
* **evaluator**：IC/RankIC/MI/退化/对齐/剪枝/消融/秩归一。
"""

from __future__ import annotations

from typing import Any

import numpy as np

# 全局随机种子（与 AppConfig 默认 seed 一致）
SEED = 20260910
DTYPE = np.float32

#: VM 公式用例（覆盖四类判定）
VM_FORMULAS: dict[str, list[int]] = {
    "xauusd": [33, 62, 3, 87, 72, 119, 73, 103],  # best_XAUUSD 真实公式
    "add": [0, 1, 65],  # ADD(f0, f1)
    "gate3": [1, 2, 3, 72],  # GATE(f1, f2, f3)（arity 3）
    "deep": [0, 1, 65, 2, 68, 74, 109],  # DIV→DECAY→CS_RANK 链
    "cs_rank": [0, 109],  # CS_RANK(f0)
    "insufficient": [65],  # 参数不足（ADD 需 2 个操作数）
    "leftover": [0, 1],  # 栈深非法（残留 2）
    "empty": [],  # 空公式 → None
    "unknown": [200],  # 未知算子 token → None
    "nan_decay": [0, 74],  # DECAY(inf) → nan_to_num
    "nan_div": [0, 1, 68],  # inf/(-inf) → nan → nan_to_num
}


def _rand_panel(rng: np.random.Generator, n: int, t: int, f: int = 65) -> np.ndarray:
    return rng.normal(0.0, 1.0, size=(n, f, t)).astype(DTYPE)


def build_feat_panels() -> dict[str, np.ndarray]:
    """构造 VM 输入面板（特征张量 ``[N, F, T]``）。"""
    panels: dict[str, np.ndarray] = {}
    panels["cs"] = _rand_panel(np.random.default_rng(SEED), 4, 320)
    panels["s_short"] = _rand_panel(np.random.default_rng(SEED + 1), 1, 300)
    panels["s_long"] = _rand_panel(np.random.default_rng(SEED + 2), 1, 800)
    panels["cs_long"] = _rand_panel(np.random.default_rng(SEED + 3), 3, 600)
    panels["const"] = np.full((2, 65, 320), 0.5, dtype=DTYPE)
    # 截断：仅 10 个特征通道（触发 `token >= feat_tensor.shape[1]` 分支）
    panels["trunc"] = _rand_panel(np.random.default_rng(SEED + 4), 1, 300, f=10)

    # NaN/Inf 面板：channel 0 = +inf，channel 1 = -inf，channel 2 = nan（其余正常）
    naninf = _rand_panel(np.random.default_rng(SEED + 5), 1, 600)
    naninf[:, 0, :] = np.inf
    naninf[:, 1, :] = -np.inf
    naninf[:, 2, :] = np.nan
    panels["naninf"] = naninf
    return panels


def build_normalize_arrays() -> dict[str, np.ndarray]:
    """构造 ``_normalize_output`` 对拍数组（覆盖滚动 / expanding / 截面 / 常数）。"""
    arrs: dict[str, np.ndarray] = {}
    arrs["n1_t120"] = np.random.default_rng(SEED + 10).normal(0, 1, (1, 120)).astype(DTYPE)
    arrs["n1_t499"] = np.random.default_rng(SEED + 11).normal(0, 1, (1, 499)).astype(DTYPE)
    arrs["n1_t500"] = np.random.default_rng(SEED + 12).normal(0, 1, (1, 500)).astype(DTYPE)
    arrs["n1_t800"] = np.random.default_rng(SEED + 13).normal(0, 1, (1, 800)).astype(DTYPE)
    arrs["n4_t320"] = np.random.default_rng(SEED + 14).normal(0, 1, (4, 320)).astype(DTYPE)
    arrs["const_n1"] = np.full((1, 300), 0.7, dtype=DTYPE)
    arrs["const_n2"] = np.full((2, 200), -0.3, dtype=DTYPE)
    return arrs


def build_signal_factors() -> dict[str, np.ndarray]:
    """构造 signal 输入因子（tanh 前）。

    中性带边界不用「恰好等于阈值」的值（``tanh`` 在 torch / numpy 间可能有 1 ULP 差异，
    会翻转 ``|pos| >= 0.05`` 的判定）；改用阈值**两侧留有 1e-4 余量**的仓位值，
    再经 ``arctanh`` 反求因子，从而稳定地覆盖「带内→0 / 带外→保留」两类逐点一致。
    """
    f: dict[str, np.ndarray] = {}
    f["n1_t600"] = np.random.default_rng(SEED + 20).normal(0, 1, (1, 600)).astype(DTYPE)
    f["n4_t320"] = np.random.default_rng(SEED + 21).normal(0, 1, (4, 320)).astype(DTYPE)
    # 目标仓位（tanh 后）取值：0 / 带宽内侧 / 带外两档 / 饱和
    target_pos = np.array(
        [0.0, 0.02, 0.0499, 0.0501, 0.06, 0.3, 0.999, -0.02, -0.0499, -0.0501, -0.06, -0.3, -0.999],
        dtype=np.float64,
    )
    edge = np.arctanh(target_pos).astype(DTYPE).reshape(1, -1)
    f["edge"] = np.tile(edge, (1, 60))
    return f


def build_backtest_cases() -> dict[str, dict[str, np.ndarray]]:
    """构造回测输入：``{case: {"factors":[N,T], "target_ret":[N,T]}}``。"""
    cases: dict[str, dict[str, np.ndarray]] = {}
    rng = np.random.default_rng(SEED + 30)
    for name, n, t in (("single", 1, 600), ("multi5", 5, 600), ("short", 1, 200)):
        factors = rng.normal(0, 1, (n, t)).astype(DTYPE)
        target_ret = (rng.normal(0, 1e-3, (n, t))).astype(DTYPE)
        cases[name] = {"factors": factors, "target_ret": target_ret}
    return cases


def build_evaluator_cases() -> dict[str, Any]:
    """构造评估器输入：候选集合 + target（含一个退化候选）。"""
    rng = np.random.default_rng(SEED + 40)
    n, t = 3, 400
    target = rng.normal(0, 1e-3, (n, t)).astype(DTYPE)
    # 让部分候选与 target 有可控相关，便于 IC/RankIC 非退化
    candidates: dict[str, np.ndarray] = {
        "A": (target * 5.0 + rng.normal(0, 1e-3, (n, t))).astype(DTYPE),
        "B": rng.normal(0, 1, (n, t)).astype(DTYPE),
        "C": (-target * 3.0 + rng.normal(0, 1e-3, (n, t))).astype(DTYPE),
        "D": (target * 4.9 + rng.normal(0, 1e-3, (n, t))).astype(DTYPE),  # 与 A 高相关
        "DEGEN": np.full((n, t), 0.25, dtype=DTYPE),  # 退化（常数）
    }
    return {
        "candidates": candidates,
        "target": target,
        "categories": {"A": "trend", "B": "momentum", "C": "reversal", "D": "trend"},
        "horizon": 2,
    }


def vm_case_keys() -> list[tuple[str, str]]:
    """返回所有 ``(panel, formula)`` 组合键。"""
    return [(p, f) for p in build_feat_panels() for f in VM_FORMULAS]
