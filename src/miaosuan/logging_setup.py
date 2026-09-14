# -*- coding: utf-8 -*-
# Copyright (c) 2026 Original Author
# Copyright (c) 2026 Your Company
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""结构化日志（架构 §9.5）。

约定：

* 日志以 **JSON 行**输出（生产环境 ``json_lines=True``），每行包含 ``run_id``；
* ``run_id`` 为一次 CLI 调用的 UUID4，贯穿该次运行的全部日志，便于聚合检索；
* 本模块**不在 import 时**创建 handler 或读取环境变量；一切通过
  :func:`configure_logging` 显式配置；
* 提供 :func:`get_logger` 获取带默认字段的 logger，**不**做全局 ``basicConfig`` 污染。

:class:`JsonLineFormatter` 可独立复用于库内部 logger。
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from datetime import UTC, datetime
from typing import Any

from .config import LoggingConfig
from .errors import MiaoSuanError

__all__ = [
    "JsonLineFormatter",
    "new_run_id",
    "configure_logging",
    "get_logger",
]

# 附加上下文字段：格式化时从 LogRecord 取出并合并进 JSON
_EXTRA_FIELDS: tuple[str, ...] = ("run_id", "symbol", "timeframe", "event", "exc_code")

_RESERVED: frozenset[str] = frozenset(
    {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "taskName",
        "message",
        "asctime",
    }
)


def new_run_id() -> str:
    """生成一次运行的唯一标识（UUID4 十六进制）。"""
    return uuid.uuid4().hex


class JsonLineFormatter(logging.Formatter):
    """把日志记录渲染为单行 JSON。

    :param run_id: 注入到每一行的运行标识；``None`` 时不写入该字段。
    """

    def __init__(self, run_id: str | None = None) -> None:
        super().__init__()
        self.run_id = run_id

    def format(self, record: logging.LogRecord) -> str:
        record.message = record.getMessage()
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.message,
        }
        if self.run_id is not None:
            payload["run_id"] = self.run_id

        # 显式附加字段
        for key in _EXTRA_FIELDS:
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value

        # 其余自定义 extra（排除保留字段）
        for key, value in record.__dict__.items():
            if key in _RESERVED or key in payload or key.startswith("_"):
                continue
            if key in _EXTRA_FIELDS:
                continue
            payload[key] = _safe(value)

        # 异常信息
        if record.exc_info:
            exc = record.exc_info[1]
            payload["exc_type"] = type(exc).__name__ if exc else "Exception"
            if isinstance(exc, MiaoSuanError):
                payload["exc_code"] = exc.code
            payload["exc_msg"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False, default=_safe)


def _safe(value: Any) -> Any:
    """把不可直接 JSON 序列化的对象降级为字符串。"""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    return repr(value)


def configure_logging(
    config: LoggingConfig | None = None,
    *,
    run_id: str | None = None,
    stream: Any | None = None,
) -> logging.Logger:
    """配置并返回妙算根 logger（``miaosuan``）。

    幂等：重复调用会清空既有 handler 后重建，避免重复输出。

    :param config: 日志配置；``None`` 时用默认 :class:`LoggingConfig`。
    :param run_id: 运行标识；``None`` 时自动生成。
    :param stream: 输出流；``None`` 时用 ``sys.stderr``（不污染 stdout 的结构化输出）。
    :returns: 已配置的 ``miaosuan`` logger。
    """
    cfg = config or LoggingConfig()
    rid = run_id or new_run_id()
    logger = logging.getLogger("miaosuan")
    logger.setLevel(_level_of(cfg.level))
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    if cfg.json_lines:
        handler.setFormatter(JsonLineFormatter(run_id=rid))
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
    logger.addHandler(handler)
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """获取带命名空间的 logger，不修改任何全局配置。"""
    return logging.getLogger("miaosuan" if not name else f"miaosuan.{name}")


def _level_of(name: str) -> int:
    level = logging.getLevelName(name.upper())
    return level if isinstance(level, int) else logging.INFO
