# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for full license text.

"""飞书群机器人通知器（路线图 R5.2 的第一个 NotifierPort 实现）。

只发文字消息（text），脱敏口径：**只发信号文字，绝不携带任何持仓金额**
（与 ``scheduler.log_bar_event`` 的转折推送共用同一正文）。

可靠性约定：``notify`` **永不抛错**——Webhook 网络失败 / 非 2xx / 超时一律
记录日志后静默返回，绝不把通知故障传导进调度回路（通知是旁路观测，不是
关键路径）。Webhook 地址**只来自配置**（``settings.yaml → notifiers.feishu.
webhook_url``），无默认地址、无环境变量回退。
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

from ..ports import NotifierPort

__all__ = ["FeishuNotifier"]

logger = logging.getLogger("miaosuan.runtime.notify.feishu")

#: Webhook 请求超时（秒）；网络抖动时快速失败，不阻塞调度循环。
_TIMEOUT_S = 5.0


class FeishuNotifier(NotifierPort):
    """把通知以 text 消息 POST 到飞书群机器人 Webhook。"""

    def __init__(self, webhook_url: str) -> None:
        if not webhook_url or not webhook_url.strip():
            raise ValueError("feishu webhook_url 不能为空（空串应走 LogNotifier 兜底）")
        self._webhook_url = webhook_url.strip()

    @property
    def webhook_url(self) -> str:
        return self._webhook_url

    def notify(self, title: str, body: str, *, level: str = "info") -> None:
        """POST 一条飞书 text 消息；任何失败只记日志，不抛出。

        ``level`` 只影响日志级别（飞书消息体不带级别语义）。
        """
        text = f"{title}\n{body}"
        payload = {"msg_type": "text", "content": {"text": text}}
        request = urllib.request.Request(
            self._webhook_url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as response:  # noqa: S310 - 地址来自配置且仅外发消息
                status = int(getattr(response, "status", 0) or 0)
                raw = response.read()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
            logger.warning("飞书通知失败（已吞）: %s", exc)
            return
        # 飞书 Webhook 对内容做风控，非 2xx 或返回非 0 code 时同样静默。
        if status and status >= 400:
            logger.warning("飞书通知被拒: HTTP %s（已吞）", status)
            return
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            logger.warning("飞书响应非 JSON（已吞）: %r", raw[:120])
            return
        if data.get("code") not in (0, None):
            logger.warning("飞书返回业务错误: %s（已吞）", data.get("msg"))