# -*- coding: utf-8 -*-
# Copyright (c) 2026 Original Author
# Copyright (c) 2026 Your Company
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""门禁综合判定（架构 §6.4，M12）。

把四项证据聚合成一个「可上线 / 仅研究 / 拦截」的结论：

==================  =========================================================
证据                 含义
==================  =========================================================
``decay_ratio``     验证分 / 训练分（衰减比）——过低 = 过拟合
``dsr``             多重检验校正后的显著性概率（见 ``multiple_testing``）
``sharpe_2x``       2x 成本下的 Sharpe —— ≤0 = 死在成本上
``wf_win_rate``     各 Walk-Forward 折验证分为正的占比 —— 稳定性
==================  =========================================================

判定：

* **硬失败 → ``BLOCKED``**：``val_score ≤ 0`` / ``sharpe_2x ≤ 0`` / ``dsr < 0.5``；
* **软失败 → ``RESEARCH_ONLY``**：``decay_ratio < 0.5`` / ``wf_win_rate < 0.6`` / ``dsr < 0.95``；
* 否则 → ``DEPLOYABLE``。

``GateVerdict`` 为 **frozen dataclass**，``to_dict`` / ``snapshot`` 产出**确定性**快照
（键排序 + 浮点定标），便于「固定 fixture → 稳定 verdict」的回归断言（验收 #4）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

__all__ = [
    "STATUS_BLOCKED",
    "STATUS_DEPLOYABLE",
    "STATUS_RESEARCH_ONLY",
    "GateThresholds",
    "GateVerdict",
    "evaluate_verdict",
]

STATUS_DEPLOYABLE = "DEPLOYABLE"
STATUS_RESEARCH_ONLY = "RESEARCH_ONLY"
STATUS_BLOCKED = "BLOCKED"

#: 快照浮点定标（小数位）。
_SNAPSHOT_NDIGITS = 6


@dataclass(frozen=True)
class GateThresholds:
    """门禁阈值（可覆盖；默认值即 M12 契约）。

    :param min_val_score: 验证分硬下界（默认 0，即须为正）。
    :param min_sharpe_2x: 2x 成本 Sharpe 硬下界（默认 0，即须为正）。
    :param min_dsr_hard: DSR 硬下界（默认 0.5，低于此即 BLOCKED）。
    :param min_dsr_soft: DSR 软下界（默认 0.95，低于此即 RESEARCH_ONLY）。
    :param min_decay_ratio: 衰减比软下界（默认 0.5）。
    :param min_wf_win_rate: Walk-Forward 胜率软下界（默认 0.6）。
    """

    min_val_score: float = 0.0
    min_sharpe_2x: float = 0.0
    min_dsr_hard: float = 0.5
    min_dsr_soft: float = 0.95
    min_decay_ratio: float = 0.5
    min_wf_win_rate: float = 0.6


@dataclass(frozen=True)
class GateVerdict:
    """门禁结论（含全部证据与失败原因，可确定性快照）。

    :param status: ``DEPLOYABLE`` / ``RESEARCH_ONLY`` / ``BLOCKED``。
    :param val_score: 开发集验证分。
    :param train_score: 开发集训练分。
    :param decay_ratio: 衰减比（``val/train``）。
    :param dsr: Deflated Sharpe 显著性概率。
    :param sharpe_1x: 1x 成本 Sharpe。
    :param sharpe_2x: 2x 成本 Sharpe。
    :param wf_win_rate: Walk-Forward 折验证分 > 0 的占比。
    :param hard_failures: 触发的硬失败项（``status=BLOCKED`` 的原因）。
    :param soft_failures: 触发的软失败项（``status=RESEARCH_ONLY`` 的原因）。
    :param n_trials: 参与多重检验的试验数。
    :param n_obs: 观测 bar 数。
    """

    status: str
    val_score: float
    train_score: float
    decay_ratio: float
    dsr: float
    sharpe_1x: float
    sharpe_2x: float
    wf_win_rate: float
    hard_failures: tuple[str, ...] = field(default_factory=tuple)
    soft_failures: tuple[str, ...] = field(default_factory=tuple)
    n_trials: int = 0
    n_obs: int = 0

    @property
    def deployable(self) -> bool:
        """是否可上线。"""
        return self.status == STATUS_DEPLOYABLE

    def to_dict(self) -> dict[str, object]:
        """JSON 友好、**确定性**（定标 + 元组转列表）的快照字典。"""
        return {
            "status": self.status,
            "val_score": round(float(self.val_score), _SNAPSHOT_NDIGITS),
            "train_score": round(float(self.train_score), _SNAPSHOT_NDIGITS),
            "decay_ratio": round(float(self.decay_ratio), _SNAPSHOT_NDIGITS),
            "dsr": round(float(self.dsr), _SNAPSHOT_NDIGITS),
            "sharpe_1x": round(float(self.sharpe_1x), _SNAPSHOT_NDIGITS),
            "sharpe_2x": round(float(self.sharpe_2x), _SNAPSHOT_NDIGITS),
            "wf_win_rate": round(float(self.wf_win_rate), _SNAPSHOT_NDIGITS),
            "hard_failures": list(self.hard_failures),
            "soft_failures": list(self.soft_failures),
            "n_trials": int(self.n_trials),
            "n_obs": int(self.n_obs),
        }

    def snapshot(self) -> str:
        """确定性 JSON 字符串（键排序 + 定标），用于回归断言。"""
        return json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)


def _safe_decay_ratio(val_score: float, train_score: float) -> float:
    """衰减比 ``val/train``；训练分非正时保守取 1.0（无衰减证据）/ 0.0。"""
    if train_score > 1e-9:
        return float(val_score / train_score)
    return 1.0 if val_score > 0 else 0.0


def evaluate_verdict(
    *,
    val_score: float,
    train_score: float,
    dsr: float,
    sharpe_1x: float,
    sharpe_2x: float,
    wf_win_rate: float,
    n_trials: int = 0,
    n_obs: int = 0,
    thresholds: GateThresholds | None = None,
) -> GateVerdict:
    """按 M12 契约聚合四项证据，返回 :class:`GateVerdict`。"""
    th = thresholds if thresholds is not None else GateThresholds()
    decay_ratio = _safe_decay_ratio(float(val_score), float(train_score))

    hard: list[str] = []
    if float(val_score) <= float(th.min_val_score):
        hard.append("val_score<=0")
    if float(sharpe_2x) <= float(th.min_sharpe_2x):
        hard.append("sharpe_2x<=0")
    if float(dsr) < float(th.min_dsr_hard):
        hard.append("dsr<0.5")

    soft: list[str] = []
    if not hard:
        if decay_ratio < float(th.min_decay_ratio):
            soft.append("decay_ratio<0.5")
        if float(wf_win_rate) < float(th.min_wf_win_rate):
            soft.append("wf_win_rate<0.6")
        if float(dsr) < float(th.min_dsr_soft):
            soft.append("dsr<0.95")

    if hard:
        status = STATUS_BLOCKED
    elif soft:
        status = STATUS_RESEARCH_ONLY
    else:
        status = STATUS_DEPLOYABLE

    return GateVerdict(
        status=status,
        val_score=float(val_score),
        train_score=float(train_score),
        decay_ratio=float(decay_ratio),
        dsr=float(dsr),
        sharpe_1x=float(sharpe_1x),
        sharpe_2x=float(sharpe_2x),
        wf_win_rate=float(wf_win_rate),
        hard_failures=tuple(hard),
        soft_failures=tuple(soft),
        n_trials=int(n_trials),
        n_obs=int(n_obs),
    )
