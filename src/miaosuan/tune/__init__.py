# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""参数寻优模块（模式 B —— 模板参数调优）。

工作流
------
1. 用户传入 ``SpecPayload``（已导出的策略 spec JSON）+ ``ParamSpace`` 列表；
2. :class:`TuneEngine` 使用 Optuna TPE 采样，在优化预算内遍历参数组合；
3. 每次试验调用 :func:`evaluate_trial` 执行全样本回测，返回 ``TuneResult``；
4. 最终按 ``n_trials`` 限制收敛，返回最优参数集。

与挖掘（模式 A）的耦合点
-------------------------
* 模式 A 产出 ``FactorPayload``；模式 B 产出 ``ParamPayload``；
* 两者共用 ``Semantics`` / ``Evidence`` / ``Provenance`` 结构，便于对比。
"""

from __future__ import annotations

from .engine import (
    TrialResult,
    TuneConfig,
    TuneEngine,
    TuneResult,
    evaluate_trial,
    load_param_space_from_spec,
    save_tune_result,
    score_trial,
)

__all__ = [
    "TrialResult",
    "TuneConfig",
    "TuneEngine",
    "TuneResult",
    "evaluate_trial",
    "load_param_space_from_spec",
    "save_tune_result",
    "score_trial",
]
