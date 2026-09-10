"""搜索层（``search/``）—— RPN-GA 因子搜索（架构 §1.3，M10–M11）。

子模块：

* :mod:`miaosuan.search.rpn` —— 定长 RPN 表示 + 栈深可行性 + 遗传算子（M10）；
* :mod:`miaosuan.search.budget` —— 三档预算 + 墙钟硬约束 + 分阶段早停（M11）；
* :mod:`miaosuan.search.islands` —— 岛屿模型（分岛 + 环形迁移，M10）；
* :mod:`miaosuan.search.ga` —— :class:`~miaosuan.search.ga.RpnGA` 进化引擎 + AM 口径适应度（M10/M11）；
* :mod:`miaosuan.search.mine` —— ``mine`` 端到端编排：数据 → 切分 → 搜索 → 门禁 → 候选（验收 #1）。

依赖方向：``search/`` 依赖 ``core/``（表示/算子/回测）与 ``config``；**绝不**被 ``core/`` 反向依赖。
``search/`` 不硬编码任何年化常数 —— 年化因子一律由 :class:`~miaosuan.market.profiles.FrozenMarketProfile`
的 ``bars_per_year`` 注入（治「魔法数字 6240」）。
"""

from __future__ import annotations

__all__: list[str] = []
