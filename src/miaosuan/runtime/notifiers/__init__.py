# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for full license text.

"""runtime 通知实现包（log 兜底；R5 追加 feishu）。"""

from __future__ import annotations

from .log import LogNotifier

__all__ = ["LogNotifier"]
