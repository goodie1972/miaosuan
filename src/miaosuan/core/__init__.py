# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""妙算核心域（``core/``）—— **纯 numpy 内核**。

铁律（CI 强制，架构 §1.2 / §9.7）：

* **不得** ``import torch``（本项目彻底去 torch）；
* **不得**读取环境变量、**不得**做文件 IO（配置与数据全部参数注入）；
* **不得** import ``adapters/``、``tune/``、``cli.py``（禁止反向依赖）。

子模块：

* :mod:`miaosuan.core.registry` —— 声明式注册层；
* :mod:`miaosuan.core.features` —— 特征注册（65 项）；
* :mod:`miaosuan.core.ops` —— 算子注册（62 项）；
* :mod:`miaosuan.core.vocab` —— 词表与确定性版本派生（移植自 AM，近乎原样）；
* :mod:`miaosuan.core.ports` —— 平台/产品无关协议（MarketProfile / DataSource / TargetPort）；
* :mod:`miaosuan.core.vm` —— 前缀公式栈式虚拟机（StackVM + 感染模型校验）；
* :mod:`miaosuan.core.evaluator` —— 因子有效性评估（IC/RankIC/Score/退化检测/_align_causal）；
* :mod:`miaosuan.core.backtest` —— 向量化组合回测（多目标 Reward）；
* :mod:`miaosuan.core.signal` —— 连续仓位信号映射（tanh + long_only 开关）。
"""

from __future__ import annotations

__all__: list[str] = []
