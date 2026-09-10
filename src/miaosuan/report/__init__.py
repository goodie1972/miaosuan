"""妙算产物层（``report/``）—— 报告持久化与指标可视化（IO 允许）。

子模块：

* :mod:`miaosuan.report.persist` —— 报告 JSON 持久化 / 反序列化（自 ``core/evaluator.py``
  拆出，满足「``core/`` 无文件 IO」铁律）。

T02 将新增 ``report/metrics.py``（统一指标 + 成本敏感性曲线）。
"""

from __future__ import annotations

__all__: list[str] = []
