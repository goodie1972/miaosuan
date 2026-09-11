"""magic 号段账本（M15）—— 6 位 magic 的分配与去重。

号段规则：``66`` + 两位序号 + 两位版本号（如 ``661801`` = 序号 18 / 版本 1）。

设计为**账本**而非「取最大值 +1」，原因：

* 已占用的号（``661601`` / ``661701``）必须永久回避，哪怕策略文件已删除；
* 同一 ``(名称, 版本)`` 重复导出必须**幂等**地返回同一个 magic，
  否则 CLI 重跑会污染号段；
* 账本落盘后可由人工审阅 / 回滚。

账本默认落在 ``<仓库根>/artifacts/magic_registry.json``（可由调用方注入路径，
测试用 ``tmp_path``）。写盘动作由 :func:`allocate_magic` 内部完成——这是平台的
**持久化状态**，不是临时文件；其余临时产物仍只在 ``artifacts/``。
"""

from __future__ import annotations

import contextlib
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

# 跨进程互斥：写入账本前先拿文件锁。msvcrt 仅 Windows 可用（本平台是 MT4/Windows），
# 非 Windows 环境降级为无锁（并发分配保护失效，但单进程与测试不受影响）。
try:  # pragma: no cover - 平台相关
    import msvcrt

    _HAS_MSVC = True
except ImportError:  # pragma: no cover - 平台相关
    msvcrt = None  # type: ignore[assignment]
    _HAS_MSVC = False

from .contract import KNOWN_MAGICS, MAGIC_LENGTH, MAGIC_PREFIX, MIN_SEQ

__all__ = [
    "DEFAULT_LEDGER_PATH",
    "MagicLedger",
    "allocate_magic",
    "known_magics",
]

#: 默认账本路径（相对仓库根：``src/miaosuan/adapters/algoforge`` 往上 5 层）。
DEFAULT_LEDGER_PATH: Path = (
    Path(__file__).resolve().parents[4] / "artifacts" / "magic_registry.json"
)

#: 序号上限（两位十进制）。
_MAX_SEQ: int = 99

#: 版本号上限（两位十进制）。
_MAX_VERSION: int = 99


@contextlib.contextmanager
def _ledger_lock(path: Path):
    """对账本加文件级互斥锁，使「加载→分配→落盘」构成原子临界区。

    两个进程若同时冷启动（都看到空账本），无锁时会各自分配到同一个 magic
    并丢失更新；持锁后串行化整个分配流程，且锁内重新加载账本，杜绝碰撞。
    """
    lock_path = path.with_name(path.name + ".lock")
    if not _HAS_MSVC:
        yield
        return
    # 账本父目录可能尚不存在（冷启动），锁文件必须随之创建。
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(lock_path, "a+")  # noqa: SIM115 - 需长期持有文件句柄
    try:
        # LK_LOCK 超时（约 10s）后抛 OSError，这里做退避重试，最长约 ~20s。
        for _ in range(200):
            try:
                msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
                break
            except OSError:
                time.sleep(0.1)
        else:
            raise RuntimeError(f"获取 magic 账本锁超时：{lock_path}")  # pragma: no cover
        yield
    finally:
        try:
            msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:  # pragma: no cover
            pass
        fh.close()


def known_magics() -> dict[str, str]:
    """返回内置已占用 magic 的副本（历史资产，禁止复用）。"""
    return dict(KNOWN_MAGICS)


class MagicLedger:
    """magic 账本：加载 → 分配 → 落盘。

    Attributes:
        path: 账本 JSON 路径。
    """

    def __init__(self, path: str | Path | None = None) -> None:
        """初始化并加载账本（文件不存在时视为空账本）。"""
        self.path = Path(path) if path is not None else DEFAULT_LEDGER_PATH
        self._records: list[dict[str, Any]] = self._load()

    def _load(self) -> list[dict[str, Any]]:
        """从磁盘加载账本；缺失 / 损坏时返回空列表（不抛异常）。"""
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        records = data.get("records") if isinstance(data, dict) else data
        if not isinstance(records, list):
            return []
        return [r for r in records if isinstance(r, dict) and "magic" in r]

    @property
    def records(self) -> tuple[dict[str, Any], ...]:
        """当前账本记录（只读视图）。"""
        return tuple(self._records)

    @property
    def used(self) -> frozenset[str]:
        """所有已占用 magic（内置已占用 ∪ 账本记录）。"""
        return frozenset(KNOWN_MAGICS) | {str(r["magic"]) for r in self._records}

    def find(self, name: str, version: int) -> str | None:
        """按 ``(名称, 版本)`` 查已分配的 magic（幂等分配的基础）。

        Args:
            name: 策略名。
            version: 版本号。

        Returns:
            已分配则返回 magic，否则 ``None``。
        """
        for record in self._records:
            if str(record.get("name", "")) == name and int(record.get("version", 0)) == version:
                return str(record["magic"])
        return None

    def allocate(
        self,
        name: str,
        version: int = 1,
        *,
        allocated_at: str = "",
        note: str = "",
    ) -> str:
        """分配 magic（同 ``(名称, 版本)`` 幂等）。

        Args:
            name: 策略名。
            version: 版本号，``[1, 99]``。
            allocated_at: 分配时刻（ISO 8601）；留空则不写该字段。
            note: 备注说明。

        Returns:
            6 位 magic 字符串。

        Raises:
            ValueError: 版本号越界；或号段耗尽（序号 > 99）。
        """
        if not 1 <= int(version) <= _MAX_VERSION:
            raise ValueError(f"version 必须落在 [1, {_MAX_VERSION}]，实际 {version!r}")
        with _ledger_lock(self.path):
            # 锁内重新加载磁盘账本，并与内存中尚未落盘的记录**合并**（磁盘优先）：
            # 既看到并发对手已写入的号，也不丢失调用方在内存里预填的记录（如测试
            # 直接构造满账本验证号段耗尽的场景）。
            disk = self._load()
            merged: dict[str, dict[str, Any]] = {str(r["magic"]): r for r in disk}
            for record_mem in self._records:
                merged.setdefault(str(record_mem["magic"]), record_mem)
            self._records = list(merged.values())
            existing = self.find(name, int(version))
            if existing is not None:
                return existing

            used = self.used
            suffix = f"{int(version):02d}"
            for seq in range(MIN_SEQ, _MAX_SEQ + 1):
                candidate = f"{MAGIC_PREFIX}{seq:02d}{suffix}"
                if len(candidate) != MAGIC_LENGTH:
                    continue
                if candidate in used:
                    continue
                record: dict[str, Any] = {
                    "magic": candidate,
                    "name": name,
                    "version": int(version),
                }
                if allocated_at:
                    record["allocated_at"] = allocated_at
                if note:
                    record["note"] = note
                self._records.append(record)
                self._save()
                return candidate
        raise ValueError(f"magic 号段耗尽：66 段 {MIN_SEQ:02d}~{_MAX_SEQ:02d} 序号已全部占用")

    def _save(self) -> None:
        """把账本写回磁盘（键排序，保证 diff 友好；父目录自动创建）。"""
        if self.path.parent and not self.path.parent.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "version": 1,
            "prefix": MAGIC_PREFIX,
            "records": sorted(self._records, key=lambda r: str(r["magic"])),
        }
        self.path.write_text(
            json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def allocate_magic(
    name: str,
    version: int = 1,
    *,
    ledger_path: str | Path | None = None,
    allocated_at: str = "",
    note: str = "",
) -> str:
    """便捷入口：加载账本 → 分配 → 落盘。

    Args:
        name: 策略名。
        version: 版本号。
        ledger_path: 账本路径；``None`` 用 :data:`DEFAULT_LEDGER_PATH`。
        allocated_at: 分配时刻。
        note: 备注。

    Returns:
        6 位 magic 字符串。
    """
    return MagicLedger(ledger_path).allocate(
        name, version, allocated_at=allocated_at, note=note
    )


def describe(magic: str, ledger_path: str | Path | None = None) -> Mapping[str, Any]:
    """查询 magic 的归属说明。

    Args:
        magic: 6 位 magic。
        ledger_path: 账本路径；``None`` 用默认路径。

    Returns:
        ``{"magic", "name", "version", "source"}``；未登记时 ``source`` 为
        ``"free"`` 或 ``"builtin"``。
    """
    if magic in KNOWN_MAGICS:
        return {
            "magic": magic,
            "name": KNOWN_MAGICS[magic],
            "version": int(magic[4:]),
            "source": "builtin",
        }
    for record in MagicLedger(ledger_path).records:
        if str(record.get("magic")) == magic:
            return {
                "magic": magic,
                "name": str(record.get("name", "")),
                "version": int(record.get("version", 0)),
                "source": "ledger",
            }
    return {"magic": magic, "name": "", "version": int(magic[4:]), "source": "free"}
