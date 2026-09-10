"""妙算核心域（``core/``）—— **纯 numpy 内核**。

铁律（CI 强制，架构 §1.2 / §9.7）：

* **不得** ``import torch``（本项目彻底去 torch）；
* **不得**读取环境变量、**不得**做文件 IO（配置与数据全部参数注入）；
* **不得** import ``adapters/``、``tune/``、``cli.py``（禁止反向依赖）。

子模块：

* :mod:`miaosuan.core.registry` —— 声明式注册层；
* :mod:`miaosuan.core.features` —— 特征注册（65 项）；
* :mod:`miaosuan.core.ops` —— 算子注册（62 项）；
* :mod:`miaosuan.core.vocab` —— 词表与确定性版本派生（移植自 AM，近乎原样）。
"""

from __future__ import annotations

__all__: list[str] = []
