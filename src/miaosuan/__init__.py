"""妙算（MiaoSuan）—— 去 torch 的声明式因子挖掘 + 神机 策略调优引擎。

本包在 import 时**不产生任何副作用**（不读环境变量、不做文件 IO、不建连接），
保证「导入即安全」。真正的重活由显式的 ``load_config`` / ``configure_logging``
等函数或 ``cli`` 入口触发。

分层（依赖方向自上而下，禁止反向）：

    cli → pipeline → {search, tune, gate, ir, report}
                          ↓
                        core（纯 numpy，无 IO / 无环境变量 / 无平台知识）
    adapters ← pipeline（core 不得 import adapters）
"""

from __future__ import annotations

__all__ = ["__version__", "TOOL_NAME"]

# 工具标识（写入产物 provenance，架构 §3.1）
TOOL_NAME: str = "miaosuan"

# 语义化版本：MAJOR.MINOR.PATCH（架构 §9.3）
__version__: str = "0.1.0"
