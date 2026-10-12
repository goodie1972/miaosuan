# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for full license text.

"""watchlist 存储（路线图 R2.3）：品种 ↔ spec ↔ 画像 的挂载清单。

落盘 ``<artifacts>/watchlist.json``（经 :func:`miaosuan.config.artifacts_dir`，
单一 env 边界），原子写；条目只存**路径与名称**，spec 加载与安全链校验在
构建 :class:`~miaosuan.runtime.scheduler.WatchItem` 时进行（坏 spec 拒绝启动）。
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from ..config import artifacts_dir
from ..errors import MiaoSuanError

__all__ = ["WatchEntry", "WatchlistStore"]

logger = logging.getLogger(__name__)

_FILE_NAME = "watchlist.json"


@dataclass(frozen=True)
class WatchEntry:
    """一条 watch 配置（持久化形态；引擎在运行时按此装配）。"""

    symbol: str
    spec_path: str
    profile: str = ""
    timeframe: str = ""  # 缺省由 spec.semantics.timeframe 决定

    def to_dict(self) -> dict[str, str]:
        return {
            "symbol": self.symbol,
            "spec_path": self.spec_path,
            "profile": self.profile,
            "timeframe": self.timeframe,
        }


class WatchlistStore:
    """watchlist 的 JSON 持久化（原子写 + 明确错误码）。"""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or (Path(artifacts_dir()) / _FILE_NAME)

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> list[WatchEntry]:
        if not self._path.is_file():
            return []
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise MiaoSuanError(
                f"watchlist 损坏: {self._path}（{exc}）",
                context={"path": str(self._path)},
            ) from exc
        entries = data.get("entries") or []
        out: list[WatchEntry] = []
        for item in entries:
            out.append(
                WatchEntry(
                    symbol=str(item.get("symbol", "")),
                    spec_path=str(item.get("spec_path", "")),
                    profile=str(item.get("profile", "")),
                    timeframe=str(item.get("timeframe", "")),
                )
            )
        return [e for e in out if e.symbol and e.spec_path]

    def save(self, entries: list[WatchEntry]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"entries": [e.to_dict() for e in entries]}
        tmp = self._path.with_name(self._path.name + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, self._path)  # 原子替换
        logger.info("watchlist 已保存: %s（%d 条）", self._path, len(entries))

    def add(self, entry: WatchEntry) -> None:
        entries = self.load()
        entries = [e for e in entries if e.symbol != entry.symbol]  # 同品种覆盖
        entries.append(entry)
        self.save(entries)

    def remove(self, symbol: str) -> bool:
        entries = self.load()
        kept = [e for e in entries if e.symbol != symbol]
        if len(kept) == len(entries):
            return False
        self.save(kept)
        return True

    def clear(self) -> None:
        """清空所有 watchlist 条目（测试用）。"""
        self.save([])
