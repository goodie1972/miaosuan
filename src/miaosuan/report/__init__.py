# -*- coding: utf-8 -*-
# Copyright (c) 2026 Original Author
# Copyright (c) 2026 Your Company
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""妙算产物层（``report/``）—— 报告持久化与指标可视化（IO 允许）。

子模块：

* :mod:`miaosuan.report.persist` —— 报告 JSON 持久化 / 反序列化（自 ``core/evaluator.py``
  拆出，满足「``core/`` 无文件 IO」铁律）；
* :mod:`miaosuan.report.metrics` —— 统一绩效指标（Sharpe/Sortino/Calmar/MDD/换手/IC）
  与成本敏感性曲线（0.5x/1x/2x/3x）。

后续 T04 将在 ``gate/cost_curve.py`` 复用本层指标。
"""

from __future__ import annotations

__all__: list[str] = []
