# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for full license text.

"""搜索检查点与断点续挖（路线图 S1 / Sample I）。

**动机**：``mine`` 是一次性批处理——进程中断（kill -9 / 断电 / 手动强杀）即全废，
跨次运行没有分数下限保护（弱结果可覆盖既有最优，S2 解决）。检查点让长任务
（standard 2h / deep 8h）**可以安心挂机**：每隔 K 代把可复现状态原子落盘，
重启时检测到检查点自动续挖。

**可复现状态 = 种子搜索的全部输入**（同 seed 重放轨迹不变）：

* ``generation``：已完成的代数（续挖从该代继续，不重复进化）；
* ``population``：当前跨岛种群（续挖直接以其为起点，不再随机初始化）；
  每一条带 ``(tokens, train_score, val_score, birth_gen)``——**分数随检查点落盘**，
  恢复后无需重新评估（``n_evaluations`` 单调不减的载体）；
* ``elite_pool``：精英池（``(val_score, birth_gen, tokens)``，恢复后先回放再演化）；
* ``rng_state``：``numpy Generator`` 位生成器状态（保证后续随机序列无损衔接）；
* ``budget_state``：早停/移民状态（``_best_seen`` / ``_last_improve_gen`` 等）；
* ``best_score``：当前最优分（验收：恢复后最优分**不低于**中断前）。

**原子落盘**（Sample I 已修的两个坑）：

1. ``np.savez`` 对不以 ``.npz`` 结尾的文件名会**静默追加 .npz** → 传文件句柄固定落盘名；
2. 嵌套结构（rng_state / budget_state）是对象数组 → 读回必须 ``allow_pickle=True``。
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

__all__ = ["MineCheckpoint", "save_ckpt", "load_ckpt", "default_ckpt_dir"]

logger = logging.getLogger("miaosuan.search.ckpt")

#: 检查点目录名（``<artifacts>/mine_ckpt/``，经 :func:`miaosuan.config.artifacts_dir`）。
_CKPT_DIR_NAME = "mine_ckpt"
#: 单次搜索的检查点文件名。
_CKPT_FILE_NAME = "ckpt.npz"


def default_ckpt_dir(artifacts: str, *, symbol: str = "", timeframe: str = "") -> Path:
    """默认检查点目录：``<artifacts>/mine_ckpt/<symbol>_<tf>``（按标的隔离）。"""
    base = Path(artifacts) / _CKPT_DIR_NAME
    if symbol or timeframe:
        base = base / f"{symbol or 'NA'}_{timeframe or 'NA'}"
    return base


@dataclass
class MineCheckpoint:
    """一次搜索的可复现状态快照（S1）。

    :param generation: 已完成的代数（0 基；续挖从 ``generation`` 继续）。
    :param population: 当前跨岛种群；每项 ``(tokens, train_score, val_score, birth_gen)``。
    :param elite_pool: 精英池；每项 ``(val_score, birth_gen, tokens)``。
    :param rng_state: ``numpy Generator`` 位生成器状态（``bit_generator.state``）。
    :param budget_state: 预算早停/移民状态（下划线私有字段的序列化）。
    :param best_score: 当前最优 ``val_score``。
    """

    generation: int
    population: list[tuple[list[int], float, float, int]]
    elite_pool: list[tuple[float, int, list[int]]]  # (val_score, birth_gen, tokens)
    rng_state: dict[str, object]
    budget_state: dict[str, object]
    best_score: float


def save_ckpt(ckpt: MineCheckpoint, path: Path) -> None:
    """原子落盘检查点（tmp → ``replace``，防半截文件）。

    ``population`` / ``elite_pool`` 是「变长 token 列表 + 标量」的混合元组，
    必须显式转 ``object`` 数组再落盘——否则 ``np.savez`` 自动广播会在
    形状推断时抛 ``inhomogeneous shape``。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    payload = {
        "generation": int(ckpt.generation),
        "population": np.asarray(ckpt.population, dtype=object),
        "elite_pool": np.asarray(ckpt.elite_pool, dtype=object),
        "rng_state": np.asarray(ckpt.rng_state, dtype=object),
        "budget_state": np.asarray(ckpt.budget_state, dtype=object),
        "best_score": float(ckpt.best_score),
    }
    with open(tmp, "wb") as f:
        np.savez(f, **payload)
    tmp.replace(path)  # 同目录原子替换（Windows 上 replace 覆盖已存在目标）
    logger.info(
        "搜索检查点已落盘: %s（gen=%d best=%.4f）", path, ckpt.generation, ckpt.best_score
    )


def load_ckpt(path: Path) -> MineCheckpoint:
    """读取检查点（对象字段需 ``allow_pickle=True``）。

    :raises FileNotFoundError: 文件不存在。
    :raises ValueError: 文件损坏 / 缺字段（调用方按「无检查点」处理）。
    """
    if not path.is_file():
        raise FileNotFoundError(f"检查点不存在: {path}")
    raw = np.load(path, allow_pickle=True)  # noqa: S301 - 自产自读的检查点文件
    try:
        return MineCheckpoint(
            generation=int(raw["generation"]),
            population=[
                (list(t), float(tr), float(vl), int(bg))
                for t, tr, vl, bg in raw["population"]
            ],
            elite_pool=[(float(s), int(g), list(t)) for s, g, t in raw["elite_pool"]],
            rng_state=raw["rng_state"].item(),
            budget_state=raw["budget_state"].item(),
            best_score=float(raw["best_score"]),
        )
    except KeyError as exc:
        raise ValueError(f"检查点缺字段: {exc}") from exc