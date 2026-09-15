# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""``DataSource`` 实现（每产品一个，架构 §2）。

* :mod:`~miaosuan.data.sources.base` —— :class:`BaseDataSource`（协议公共骨架）；
* :mod:`~miaosuan.data.sources.parquet_mt` —— MT4/外汇 Parquet 源（MVP 唯一实现）。

新增一个市场 = 新增一个 ``DataSource`` 实现 + 一个 ``MarketProfile`` 实例 + 一份数据，
**不改 ``core/``**。
"""

from __future__ import annotations

__all__: list[str] = []
