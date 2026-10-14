# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for full license text.

"""runtime 通知实现包（log 兜底；R5 追加 feishu）。

:func:`build_notifier` 是 CLI 侧的装配入口：**只认配置**——
``settings.yaml → notifiers.feishu.webhook_url`` 非空才接飞书通道，
否则仅返回日志兜底；无默认地址、无环境变量回退。
"""

from __future__ import annotations

from ..ports import NotifierPort
from .feishu import FeishuNotifier
from .log import LogNotifier

__all__ = ["LogNotifier", "FeishuNotifier", "MultiNotifier", "build_notifier"]


class MultiNotifier(NotifierPort):
    """扇出通知器：把同一条通知同时发给多个底层通道（任一失败互不影响）。

    底层通道必须各自保证 ``notify`` 永不抛错（见 :class:`FeishuNotifier` /
    :class:`LogNotifier` 约定），本类不做额外容错。
    """

    def __init__(self, *inner: NotifierPort) -> None:
        self._inner = list(inner)

    @property
    def channels(self) -> list[NotifierPort]:
        return list(self._inner)

    def notify(self, title: str, body: str, *, level: str = "info") -> None:
        for ch in self._inner:
            ch.notify(title, body, level=level)


def build_notifier() -> NotifierPort:
    """按当前配置装配通知通道（热重载后重新调用即取新值）。

    规则：日志兜底恒在；``notifiers.feishu.webhook_url`` 非空 → 追加飞书通道。
    空串 / 未配置 → 仅日志（调用方无需判空，返回的 NotifierPort 永远可用）。
    """
    from ...settings import get_config

    cfg = get_config()
    channels: list[NotifierPort] = [LogNotifier()]
    webhook = (cfg.notifiers.feishu.webhook_url or "").strip()
    if webhook:
        channels.append(FeishuNotifier(webhook))
    return MultiNotifier(*channels) if len(channels) > 1 else channels[0]