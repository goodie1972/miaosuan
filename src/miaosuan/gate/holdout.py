# -*- coding: utf-8 -*-
# Copyright (c) 2026 Original Author
# Copyright (c) 2026 Your Company
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""hold-out 一次性封印闸（架构 §6.1 / §9.5，M12）。

**铁律**：hold-out 段是「最后一次、只此一次」的诚实检验。搜索路径（GA 循环）**绝不**触碰
hold-out，只有**最终门禁**允许消费一次。本模块把这条纪律收敛为一个显式对象
:class:`HoldoutGate`：

* :meth:`HoldoutGate.seal` —— 封印一个 hold-out 指纹（第二次抛
  :class:`~miaosuan.errors.HoldoutSealedError`）；
* :meth:`HoldoutGate.allow_evaluation` —— 「封印即消费」语义：**首次**调用封印并放行，
  之后任何同一指纹的调用一律抛错（杜绝反复 peek）；
* :meth:`HoldoutGate.sealed_count` —— 已封印指纹数（供「搜索未触碰」的证据断言）。

底层台账复用 :class:`~miaosuan.data.split.HoldoutSealRegistry`（内存或落盘）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..data.split import HoldoutSealRegistry
from ..errors import HoldoutSealedError

__all__ = ["HoldoutGate"]


@dataclass
class HoldoutGate:
    """hold-out 一次性封印闸（对 :class:`HoldoutSealRegistry` 的语义封装）。

    :param registry: 封印台账（``HoldoutSealRegistry``；测试可用 ``IN_MEMORY``）。
    """

    registry: HoldoutSealRegistry

    def is_sealed(self, fingerprint: str) -> bool:
        """该指纹是否已被封印（消费过）。"""
        return self.registry.is_sealed(fingerprint)

    def seal(self, fingerprint: str, *, meta: dict[str, Any] | None = None) -> None:
        """封印指纹；已存在则抛 :class:`HoldoutSealedError`。"""
        self.registry.seal(fingerprint, meta=meta)

    def allow_evaluation(self, fingerprint: str, *, meta: dict[str, Any] | None = None) -> bool:
        """「封印即消费」：首次调用封印并返回 ``True``；重复调用抛错。

        :raises HoldoutSealedError: 该指纹已被消费过。
        """
        self.registry.seal(fingerprint, meta=meta)
        return True

    def sealed_count(self) -> int:
        """已封印的指纹数量（用于「搜索路径未触碰」断言）。"""
        return len(self.registry.list_seals())

    def list_seals(self) -> list[str]:
        """已封印指纹列表（字典序）。"""
        return self.registry.list_seals()

    def assert_untouched(self, fingerprint: str) -> None:
        """断言某指纹**尚未**被封印（搜索路径不应触碰 hold-out）。

        :raises HoldoutSealedError: 该指纹已被封印（说明搜索路径违规触碰了 hold-out）。
        """
        if self.registry.is_sealed(fingerprint):
            raise HoldoutSealedError(
                "搜索路径违规触碰了 hold-out：该指纹已被封印",
                context={"fingerprint": fingerprint},
            )
