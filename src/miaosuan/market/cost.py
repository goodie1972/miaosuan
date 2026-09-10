"""``CostModel`` —— 交易成本模型（架构 §3.2 ``CostModel``，买卖可不对称）。

对应 :class:`~miaosuan.core.ports.CostModelProtocol`：

* ``commission`` —— 佣金率；
* ``slippage`` —— 滑点率；
* ``sell_tax`` —— 卖出税（**仅当** ``asymmetric=True`` 计入 ``sell`` 方向；A 股印花税）；
* ``total(side, notional)`` —— 某方向的**单边成本率**（``notional`` 预留给未来阶梯费率，
  当前成本率为常数，不随名义金额变化）。

本模块属 ``market/``（产品轴「市场怎么交易」），故可含具体市场的费率取值；``core/`` 不感知。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import ConfigError

__all__ = ["CostModel", "ZERO_COST"]

_VALID_SIDES = ("buy", "sell")


@dataclass(frozen=True)
class CostModel:
    """单边成本率模型（佣金 + 滑点 [+ 卖出税]）。

    :param commission: 佣金率（单边）。
    :param slippage: 滑点率（单边）。
    :param sell_tax: 卖出税（印花税等）；仅在 ``asymmetric=True`` 时对 ``sell`` 生效。
    :param asymmetric: 买卖是否不对称（A 股卖出印花税 → True）。
    """

    commission: float = 0.0
    slippage: float = 0.0
    sell_tax: float = 0.0
    asymmetric: bool = False

    def __post_init__(self) -> None:
        for field_name in ("commission", "slippage", "sell_tax"):
            value = getattr(self, field_name)
            if value < 0:
                raise ConfigError(
                    f"CostModel.{field_name} 不得为负", context={field_name: value}
                )

    # ── 协议实现 ────────────────────────────────────────────────────────────

    def total(self, side: str, notional: float = 1.0) -> float:
        """返回某方向的单边总成本率（``buy`` / ``sell``）。

        :param side: ``"buy"`` 或 ``"sell"``。
        :param notional: 名义金额（预留给阶梯费率；当前不影响结果）。
        """
        if side not in _VALID_SIDES:
            raise ConfigError(f"side 非法: {side!r}（可选 {_VALID_SIDES}）")
        return self.buy_rate() if side == "buy" else self.sell_rate()

    # ── 分量 ────────────────────────────────────────────────────────────────

    def buy_rate(self) -> float:
        """买入单边成本率（佣金 + 滑点）。"""
        return float(self.commission + self.slippage)

    def sell_rate(self) -> float:
        """卖出单边成本率（佣金 + 滑点 + 卖出税[若不对称]）。"""
        tax = self.sell_tax if self.asymmetric else 0.0
        return float(self.commission + self.slippage + tax)

    def round_trip_rate(self) -> float:
        """一次完整买卖往返的成本率（买 + 卖）。"""
        return float(self.buy_rate() + self.sell_rate())

    def scaled(self, multiplier: float) -> CostModel:
        """按倍率缩放全部成本分量（用于成本敏感性曲线）。"""
        return CostModel(
            commission=self.commission * multiplier,
            slippage=self.slippage * multiplier,
            sell_tax=self.sell_tax * multiplier,
            asymmetric=self.asymmetric,
        )


#: 零成本模型（诊断用）
ZERO_COST = CostModel()
