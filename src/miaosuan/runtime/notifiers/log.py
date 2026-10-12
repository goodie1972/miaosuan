# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for full license text.

"""日志通知器（NotifierPort 兜底实现：所有通知落结构化日志）。"""

from __future__ import annotations

import logging

from ..ports import NotifierPort

__all__ = ["LogNotifier"]

_LEVELS = {"info": logging.INFO, "warning": logging.WARNING, "error": logging.ERROR}

logger = logging.getLogger("miaosuan.runtime.notify")


class LogNotifier(NotifierPort):
    """把通知写进日志（``miaosuan.runtime.notify`` 通道）；永不抛错。"""

    def notify(self, title: str, body: str, *, level: str = "info") -> None:
        logger.log(_LEVELS.get(level, logging.INFO), "%s | %s", title, body)
