# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""门禁层（``gate/``）—— 从「搜索出高分」到「可部署」的最后一道闸（架构 §6.4，M12）。

子模块：

* :mod:`miaosuan.gate.multiple_testing` —— 多重检验校正（Deflated Sharpe + Bonferroni/BH）；
* :mod:`miaosuan.gate.cost_curve` —— 成本敏感性曲线（0.5x/1x/2x/3x）与 2x 成本闸；
* :mod:`miaosuan.gate.holdout` —— hold-out 一次性封印闸（搜索路径**不得**触碰）；
* :mod:`miaosuan.gate.verdict` —— 综合判定 :class:`~miaosuan.gate.verdict.GateVerdict`。

判定契约（M12）：

* **硬失败 → ``BLOCKED``**：验证分非正 / 2x 成本下 Sharpe 非正 / DSR 不显著（仅噪声水平）；
* **软失败 → ``RESEARCH_ONLY``**：过（可上线）但存在瑕疵 —— 衰减比过低（过拟合）、
  Walk-Forward 胜率不足、多重检验显著性未达高标准；
* 全部通过 → ``DEPLOYABLE``。

依赖方向：``gate/`` 依赖 ``core``（回测/信号）、``report.metrics``、``data.split``（封印台账）；
**绝不** import ``search``（避免搜索—门禁循环依赖）。
"""

from __future__ import annotations

__all__: list[str] = []
