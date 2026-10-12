# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for full license text.

"""runtime 状态持久化（路线图 R3.3）：账本 + 单实例锁 + kill 文件。

三个验收基石（D6 顺序：能恢复 → 能对上账 → 能急停）：

* **账本**：intents（挂单意图）+ fills（成交回执）JSON 落盘，原子写
  （tmp → os.replace）；持仓由 fills 净额聚合派生，不单独存——崩溃恢复
  = 重读账本 + 重放未完成意图（挂回 PaperBroker 队列）。
* **单实例锁**：``<name>.lock`` 写入 PID + 时间戳；活进程持锁时拒绝第二
  实例（``E-RUNTIME-STATE-LOCK``）；检测到死进程遗留的陈旧锁自动接管。
* **kill 文件**：状态目录下 ``STOP`` 文件存在即触发急停（平仓停机）——
  ``miaosuan paper stop`` 的实现就是创建该文件，运行回路每个调度周期
  检查一次（验收：kill ≤ 1 个调度周期内生效）。
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import artifacts_dir
from ..errors import RuntimeStateLockError

__all__ = [
    "Ledger",
    "LockHandle",
    "kill_file_exists",
    "kill_file_path",
    "state_dir",
]

logger = logging.getLogger(__name__)

_STATE_DIR_NAME = "runtime"
_KILL_FILE_NAME = "STOP"


def state_dir(root: Path | None = None) -> Path:
    """runtime 状态目录（``<artifacts>/runtime/``；测试可注入根目录）。"""
    base = Path(root) if root is not None else Path(artifacts_dir())
    return base / _STATE_DIR_NAME


def kill_file_path(root: Path | None = None) -> Path:
    """kill 开关文件路径（存在即急停）。"""
    return state_dir(root) / _KILL_FILE_NAME


def kill_file_exists(root: Path | None = None) -> bool:
    """STOP 文件是否存在（启动安全链与运行回路共用）。"""
    return kill_file_path(root).is_file()


def _pid_alive(pid: int) -> bool:
    """进程是否存活（Windows 安全实现）。

    ⚠ 不能用 ``os.kill(pid, 0)``：Windows 上 ``os.kill`` 对非 CTRL 信号
    走 ``TerminateProcess``——"探测"即杀死目标进程。POSIX 才有信号 0 探活
    语义；Windows 用 ``OpenProcess + GetExitCodeProcess``。
    """
    if pid <= 0:
        return True  # 无法判定时保守视为存活
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except OSError:
            return True  # 存在但无权发信号
        return True
    import ctypes

    k32 = ctypes.windll.kernel32
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    ERROR_ACCESS_DENIED = 5
    STILL_ACTIVE = 259
    k32.OpenProcess.restype = ctypes.c_void_p
    k32.OpenProcess.argtypes = (ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32)
    handle = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return bool(k32.GetLastError() == ERROR_ACCESS_DENIED)  # 存在但无权 → 活
    try:
        code = ctypes.c_uint32()
        if k32.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(code)):
            return bool(code.value == STILL_ACTIVE)
        return True
    finally:
        k32.CloseHandle(ctypes.c_void_p(handle))


@dataclass
class LockHandle:
    """单实例锁（写 PID + 时间戳；陈旧锁自动接管）。"""

    path: Path
    _acquired: bool = field(default=False, init=False)

    def acquire(self) -> None:
        """获取锁；已被活进程持有时抛 :class:`RuntimeStateLockError`。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_file():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                holder_pid = int(data.get("pid", 0))
                held_since = float(data.get("ts", 0.0))
            except (json.JSONDecodeError, OSError, ValueError):
                holder_pid, held_since = 0, 0.0
            if holder_pid == os.getpid():
                pass  # 同进程重复获取（幂等）
            elif _pid_alive(holder_pid):
                raise RuntimeStateLockError(
                    f"状态锁被占用: {self.path}（pid={holder_pid}，"
                    f"持有 {time.time() - held_since:.0f}s）"
                    "——先停掉既有实例或删除陈旧锁文件",
                    context={"path": str(self.path), "holder_pid": holder_pid},
                )
            else:
                logger.warning("接管陈旧状态锁: %s（pid=%d 已不存在）", self.path, holder_pid)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(
            json.dumps({"pid": os.getpid(), "ts": time.time()}, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(tmp, self.path)
        self._acquired = True

    def release(self) -> None:
        """释放锁（仅当仍由本进程持有时删除）。"""
        if not self._acquired:
            return
        try:
            if self.path.is_file():
                data = json.loads(self.path.read_text(encoding="utf-8"))
                if int(data.get("pid", -1)) == os.getpid():
                    self.path.unlink()
        except (json.JSONDecodeError, OSError):  # pragma: no cover - 释放尽力而为
            pass
        self._acquired = False


@dataclass(frozen=True)
class Fill:
    """一笔成交回执（账本最小记账单元；持仓 = fills 净额聚合）。"""

    fill_id: str
    intent_id: str
    symbol: str
    side: str  # buy / sell
    lots: float
    price: float
    fee: float
    ts: int  # UTC epoch 秒（成交时刻）
    magic: int
    reason: str = ""  # signal / close:<reason> / flatten:<reason>

    def to_dict(self) -> dict[str, Any]:
        return {
            "fill_id": self.fill_id,
            "intent_id": self.intent_id,
            "symbol": self.symbol,
            "side": self.side,
            "lots": self.lots,
            "price": self.price,
            "fee": self.fee,
            "ts": self.ts,
            "magic": self.magic,
            "reason": self.reason,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Fill:
        return Fill(
            fill_id=str(data["fill_id"]),
            intent_id=str(data["intent_id"]),
            symbol=str(data["symbol"]),
            side=str(data["side"]),
            lots=float(data["lots"]),
            price=float(data["price"]),
            fee=float(data["fee"]),
            ts=int(data["ts"]),
            magic=int(data["magic"]),
            reason=str(data.get("reason", "")),
        )


class Ledger:
    """纸面账本：intents + fills 的 JSON 持久化（原子写、崩溃可恢复）。

    - ``intents``：挂单意图（pending / filled / cancelled）；
    - ``fills``：成交回执；持仓由 fills 按 (symbol, magic) 净额聚合派生；
    - 每次变更立即原子落盘——进程崩溃最多丢"正在写的那一次"，账本本身
      永不半截（D6「能恢复」的载体）。
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._intents: list[dict[str, Any]] = []
        self._fills: list[Fill] = []
        self._intent_seq = 0
        self._fill_seq = 0
        self._load()

    # ── 持久化 ────────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self._path.is_file():
            return
        data = json.loads(self._path.read_text(encoding="utf-8"))
        self._intents = list(data.get("intents") or [])
        self._fills = [Fill.from_dict(d) for d in (data.get("fills") or [])]
        self._intent_seq = int(data.get("intent_seq", 0))
        self._fill_seq = int(data.get("fill_seq", 0))

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "intents": self._intents,
            "fills": [f.to_dict() for f in self._fills],
            "intent_seq": self._intent_seq,
            "fill_seq": self._fill_seq,
        }
        tmp = self._path.with_name(self._path.name + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, self._path)

    # ── intents ──────────────────────────────────────────────────────────

    def append_intent(
        self, symbol: str, side: str, lots: float, *, magic: int, comment: str = ""
    ) -> str:
        """登记挂单意图（pending），返回 intent_id。"""
        self._intent_seq += 1
        intent_id = f"i{self._intent_seq:08d}"
        self._intents.append(
            {
                "intent_id": intent_id,
                "symbol": symbol,
                "side": side,
                "lots": float(lots),
                "magic": int(magic),
                "comment": comment,
                "status": "pending",
                "created_ts": int(time.time()),
            }
        )
        self._save()
        return intent_id

    def pending_intents(self, symbol: str | None = None) -> list[dict[str, Any]]:
        """未完成意图（崩溃恢复时重放回 PaperBroker 队列）。"""
        return [
            i
            for i in self._intents
            if i.get("status") == "pending" and (symbol is None or i.get("symbol") == symbol)
        ]

    def mark_intent_status(self, intent_id: str, status: str) -> None:
        """更新意图状态（filled / cancelled）。"""
        for intent in self._intents:
            if intent.get("intent_id") == intent_id:
                intent["status"] = status
                break
        self._save()

    def intents(self) -> list[dict[str, Any]]:
        return list(self._intents)

    # ── fills ────────────────────────────────────────────────────────────

    def append_fill(
        self,
        intent_id: str,
        symbol: str,
        side: str,
        lots: float,
        price: float,
        fee: float,
        *,
        magic: int,
        reason: str = "",
        ts: int | None = None,
    ) -> str:
        """登记成交回执并落盘，返回 fill_id。"""
        self._fill_seq += 1
        fill = Fill(
            fill_id=f"f{self._fill_seq:08d}",
            intent_id=intent_id,
            symbol=symbol,
            side=side,
            lots=float(lots),
            price=float(price),
            fee=float(fee),
            ts=int(ts if ts is not None else time.time()),
            magic=int(magic),
            reason=reason,
        )
        self._fills.append(fill)
        self._save()
        return fill.fill_id

    def fills(self, since_ts: int = 0) -> list[Fill]:
        """``since_ts`` 之后的成交回执（对账与回执通道共用）。"""
        return [f for f in self._fills if f.ts > since_ts]

    def all_fills(self) -> list[Fill]:
        return list(self._fills)

    # ── 持仓（fills 净额聚合，不单独持久化）──────────────────────────────

    def open_positions(self) -> list[dict[str, Any]]:
        """按 (symbol, magic) 净额聚合的持仓快照。

        返回项：``symbol / magic / side(long|short) / lots / avg_price /
        position_id``；净额为 0 的组合不出现（已平净）。
        """
        signed: dict[tuple[str, int], list[Fill]] = {}
        for f in self._fills:
            signed.setdefault((f.symbol, f.magic), []).append(f)
        out: list[dict[str, Any]] = []
        for (symbol, magic), fills in sorted(signed.items()):
            net = sum(f.lots if f.side == "buy" else -f.lots for f in fills)
            if abs(net) < 1e-12:
                continue
            if net > 0:
                legs = [f for f in fills if f.side == "buy"]
            else:
                legs = [f for f in fills if f.side == "sell"]
            lots_total = sum(f.lots for f in legs) or 1.0
            avg = sum(f.price * f.lots for f in legs) / lots_total
            out.append(
                {
                    "symbol": symbol,
                    "magic": magic,
                    "side": "long" if net > 0 else "short",
                    "lots": round(abs(net), 8),
                    "avg_price": round(avg, 8),
                    "position_id": f"{symbol}:{magic}",
                }
            )
        return out

    def total_fees(self) -> float:
        return round(sum(f.fee for f in self._fills), 8)
