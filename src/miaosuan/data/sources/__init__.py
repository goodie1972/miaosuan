"""``DataSource`` 实现（每产品一个，架构 §2）。

* :mod:`~miaosuan.data.sources.base` —— :class:`BaseDataSource`（协议公共骨架）；
* :mod:`~miaosuan.data.sources.parquet_mt` —— MT4/外汇 Parquet 源（MVP 唯一实现）。

新增一个市场 = 新增一个 ``DataSource`` 实现 + 一个 ``MarketProfile`` 实例 + 一份数据，
**不改 ``core/``**。
"""

from __future__ import annotations

__all__: list[str] = []
