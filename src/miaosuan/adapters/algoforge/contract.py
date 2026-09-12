"""AlgoForge 平台契约常量（M15）。

集中定义**不可变**的平台约定，避免散落在生成器与模板里：

* magic 号段与已占用清单；
* 文件命名 ``YYYYMMDD_名称_vN.py``；
* 生成文件必须声明的 ``STRATEGY_*`` 常量；
* :data:`PLATFORM_SPEC` —— 交给 :mod:`miaosuan.adapters.base` 的平台描述
  （含数据驱动的 ``stub_modules``，使生成文件可在离线环境被 import）。

magic 规则（6 位十进制）：``66`` + 两位序号 + 两位版本号。``66`` 为自用号段，
与 AlgoForge 既有策略号段隔离。序号由 :mod:`magic_registry` 账本分配，**不得**
硬编码。
"""

from __future__ import annotations

from ..base import PlatformSpec

__all__ = [
    "GATE_TOKEN_NAME",
    "KNOWN_MAGICS",
    "MAGIC_LENGTH",
    "MAGIC_PREFIX",
    "MIN_SEQ",
    "PLATFORM_NAME",
    "PLATFORM_SPEC",
    "REQUIRED_CONSTANTS",
    "default_filename",
]

#: 平台名（与目录名一致）。
PLATFORM_NAME: str = "algoforge"

#: magic 号段前缀（自用）。
MAGIC_PREFIX: str = "66"

#: magic 总位数。
MAGIC_LENGTH: int = 6

#: 序号分配起点：``6616xx``（multi_confluence_quant）与 ``6617xx``（alphagate）
#: 已占用，故从 18 开始。
MIN_SEQ: int = 18

#: 已占用 magic（历史资产，禁止复用）。值仅为人类可读说明。
KNOWN_MAGICS: dict[str, str] = {
    "661601": "20260630_multi_confluence_quant_v1",
    "661701": "20260909_h1_alphagate_v1",
}

#: 生成文件必须声明的模块级常量（缺失即 lint ERROR）。
REQUIRED_CONSTANTS: tuple[str, ...] = (
    "STRATEGY_MAGIC",
    "STRATEGY_NAME",
    "STRATEGY_VERSION",
    "STRATEGY_CHANGELOG",
)

#: 触发 R1（GATE 零点不连续）保护的算子名。
GATE_TOKEN_NAME: str = "GATE"

#: AlgoForge 平台描述（数据驱动）。
PLATFORM_SPEC: PlatformSpec = PlatformSpec(
    name=PLATFORM_NAME,
    file_suffix=".py",
    module_header=(
        "import numpy as np",
        "from strategies.base import BaseStrategy",
        # OrderType：generate_signal 必须返回 OrderType 枚举（on_tick 读 .value）。
        # 刻意**不**导入 MT4BridgeBase —— 它有 9 个抽象方法，多继承会把导出类
        # 变成抽象类，平台实例化直接 TypeError（策略加载不起来）。
        "from core.bridge import OrderType",
    ),
    stub_modules={
        "strategies.base": ("BaseStrategy",),
        # MT4BridgeBase 保留在占位里（即便已不导入）：一旦有人把它加回
        # base_classes，占位会如实提供 9 个抽象方法，回归测试的
        # __abstractmethods__ 断言立刻失败，而不是等到实盘才发现。
        "core.bridge": ("MT4BridgeBase", "OrderType"),
    },
    # OrderType 是枚举：占位必须造出真 Enum，否则离线环境里 OrderType.BUY 直接
    # AttributeError，生成文件连 import 后的信号路径都跑不通。成员与真平台一致。
    stub_enums={
        "core.bridge": {
            "OrderType": ("BUY", "SELL", "BUY_LIMIT", "SELL_LIMIT", "BUY_STOP", "SELL_STOP"),
        },
    },
    # 占位基类必须如实还原抽象面（成员名逐条取自 algoforge-dev 真源码）：
    #   strategies/base.py:345  BaseStrategy  → generate_signal
    #   core/bridge.py:68-120   MT4BridgeBase → 下列 9 个桥接方法
    stub_abstracts={
        "strategies.base": {"BaseStrategy": ("generate_signal",)},
        "core.bridge": {
            "MT4BridgeBase": (
                "connect",
                "disconnect",
                "get_account_info",
                "get_positions",
                "get_candles",
                "get_tick_price",
                "open_order",
                "close_order",
                "modify_order",
            ),
        },
    },
    supports_short=True,
    # 只继承 BaseStrategy：它自带 refresh_data()（灌 self.candles）与 on_tick()，
    # 导出策略不需要 MT4BridgeBase 的任何能力；多继承它只会引入 9 个未实现的
    # 抽象方法，把类变成抽象类。AlgoForge 现网 27 个策略也全是单继承 BaseStrategy。
    base_classes=("BaseStrategy",),
    default_symbol="XAUUSD",
)


def default_filename(name: str, version: int, *, date: str = "") -> str:
    """生成 AlgoForge 约定的文件名 ``YYYYMMDD_名称_vN.py``。

    Args:
        name: 策略名（已清洗为 ``[A-Za-z0-9_]+``）。
        version: 版本号（>= 1）。
        date: ``YYYYMMDD``；留空则用当天 UTC 日期。

    Returns:
        文件名（含 ``.py`` 后缀）。
    """
    from datetime import UTC, datetime

    stamp = date or datetime.now(UTC).strftime("%Y%m%d")
    return f"{stamp}_{name}_v{int(version)}.py"
