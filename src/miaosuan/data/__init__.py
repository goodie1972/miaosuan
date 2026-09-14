# -*- coding: utf-8 -*-
# Copyright (c) 2026 Original Author
# Copyright (c) 2026 Your Company
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""数据层（``data/``）—— ★产品轴「数据怎么来」（架构 §2 / §9.4）。

职责边界（架构 §9.4 硬分界）：

* **复权 / 交易日历 / 缺失与停牌处理 / 时间索引**归本层（:mod:`~miaosuan.data.sources`）；
* **成本 / 多空 / 杠杆 / 合约规格 / 时段 / 结算**归 :mod:`miaosuan.market`；
* ``Panel`` 是两者交汇处的**纯数据容器**（无市场知识）。

子模块：

* :mod:`miaosuan.data.panel` —— OHLCV 数据契约（``[N, T]`` + time + symbols + fingerprint）；
* :mod:`miaosuan.data.fingerprint` —— sha256 数据指纹（治 R14）；
* :mod:`miaosuan.data.loader` —— Parquet / CSV → Panel；
* :mod:`miaosuan.data.split` —— 三段切分 + purge/embargo + hold-out 一次性封印；
* :mod:`miaosuan.data.sources` —— ``DataSource`` 实现（每产品一个）。

依赖方向（架构 §9.7 规则 4）：``data/`` **不得** import ``market/``。
"""

from __future__ import annotations

__all__: list[str] = []
