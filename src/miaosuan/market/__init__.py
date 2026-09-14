# -*- coding: utf-8 -*-
# Copyright (c) 2026 Original Author
# Copyright (c) 2026 Your Company
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""市场画像层（``market/``）—— ★产品轴「市场怎么交易」（架构 §2 / §9.4）。

职责边界（架构 §9.4 硬分界）：本层只管**成本 / 多空 / 杠杆 / 合约规格 / 时段 / 可交易性 / 结算**；
**复权 / 交易日历 / 缺失处理**归 :mod:`miaosuan.data.sources`。

子模块：

* :mod:`miaosuan.market.cost` —— ``CostModel``（买卖可不对称）；
* :mod:`miaosuan.market.profiles` —— ``MarketProfile`` v1 实例（``FOREX_XAUUSD`` + 3 占位）。

依赖方向（架构 §9.7 规则 4）：``market/`` **不得** import ``data/``。
"""

from __future__ import annotations

__all__: list[str] = []
