# -*- coding: utf-8 -*-
# Copyright (c) 2026 Original Author
# Copyright (c) 2026 Your Company
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""三段切分 + purge/embargo + hold-out 一次性封印（架构 §6.1，M7）。

```
|<────── Train 60% ──────>|<-purge->|<── Val 20% ──>|<-purge->|<─ Hold-out 20% ─>|
                          ↑                          ↑         ↑
                       gap 20 bars              gap 20 bars   永不参与训练/选择
```

* **purge / embargo**：段间留 ``purge_gap + embargo`` 根 bar，切断「标签重叠 / 前视泄漏」；
  hold-out 取到数据末端，保证「最近一段从未参与训练/选择」。
* **一次性封印**：hold-out 指纹写入台账 ``artifacts/holdout_seals.json``，**同一指纹只能用一次**；
  第二次用同一 hold-out → 抛 :class:`~miaosuan.errors.HoldoutSealedError`（杜绝反复 peek）。
* **可复现**：切分完全由 ``(数据指纹, 比例, purge, embargo, seed)`` 决定，与调用时机无关。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..config import SplitConfig
from ..errors import DataError, HoldoutSealedError
from .fingerprint import holdout_fingerprint, split_fingerprint
from .panel import Panel

__all__ = [
    "DEFAULT_SEAL_PATH",
    "DataSplit",
    "HoldoutSealRegistry",
    "SplitIndices",
    "make_split",
    "three_way_split",
]

#: 默认封印台账路径（``.gitignore`` 已忽略 ``artifacts/*``）
DEFAULT_SEAL_PATH = Path("artifacts") / "holdout_seals.json"

#: 台账 schema 版本
_LEDGER_VERSION = 1


@dataclass(eq=False)
class SplitIndices:
    """三段的整数索引（``[start, end)`` 连续区间）。

    :param train: 训练段索引。
    :param val: 验证段索引。
    :param holdout: hold-out 段索引（最末一段）。
    :param purge_gap: 段间 purge 间隔（bar 数）。
    :param embargo: 防标签重叠的额外间隔（bar 数）。
    :param ratios: ``(train, val, holdout)`` 目标比例。
    :param seed: 参与指纹的随机种子。
    """

    train: np.ndarray
    val: np.ndarray
    holdout: np.ndarray
    purge_gap: int
    embargo: int
    ratios: tuple[float, float, float]
    seed: int

    @property
    def gap(self) -> int:
        """段间实际间隔 = ``purge_gap + embargo``。"""
        return int(self.purge_gap + self.embargo)

    def sizes(self) -> tuple[int, int, int]:
        """返回 ``(len(train), len(val), len(holdout))``。"""
        return int(self.train.size), int(self.val.size), int(self.holdout.size)

    def ranges(self) -> dict[str, tuple[int, int]]:
        """返回三段的开闭区间 ``{段名: (start, end)}``（``end`` 端开）。"""

        def _range(arr: np.ndarray) -> tuple[int, int]:
            return int(arr[0]), int(arr[-1]) + 1

        return {"train": _range(self.train), "val": _range(self.val), "holdout": _range(self.holdout)}


def three_way_split(
    n_bars: int,
    *,
    ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
    purge_gap: int = 20,
    embargo: int = 4,
    seed: int = 0,
) -> SplitIndices:
    """按时间序把 ``n_bars`` 切成 train / val / holdout 三段（含 purge+embargo）。

    :raises DataError: 比例非法或数据过短以致某段为空。
    """
    if n_bars < 3:
        raise DataError("n_bars 过小，无法三段切分", context={"n_bars": n_bars})
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise DataError("三段比例之和必须为 1.0", context={"ratios": ratios})
    if min(ratios) < 0:
        raise DataError("三段比例不得为负", context={"ratios": ratios})
    if purge_gap < 0 or embargo < 0:
        raise DataError("purge_gap / embargo 不得为负")

    gap = int(purge_gap + embargo)
    train_size = int(n_bars * ratios[0])
    val_size = int(n_bars * ratios[1])

    train_end = train_size
    val_start = train_end + gap
    val_end = val_start + val_size
    holdout_start = val_end + gap

    if train_end < 1:
        raise DataError("训练段为空（数据过短或比例过小）", context={"n_bars": n_bars})
    if val_end <= val_start:
        raise DataError("验证段为空（数据过短或比例过小）", context={"n_bars": n_bars})
    if holdout_start >= n_bars:
        raise DataError(
            "hold-out 段为空（数据过短或比例过小）",
            context={"n_bars": n_bars, "holdout_start": holdout_start},
        )

    return SplitIndices(
        train=np.arange(0, train_end, dtype=np.int64),
        val=np.arange(val_start, val_end, dtype=np.int64),
        holdout=np.arange(holdout_start, n_bars, dtype=np.int64),
        purge_gap=int(purge_gap),
        embargo=int(embargo),
        ratios=(float(ratios[0]), float(ratios[1]), float(ratios[2])),
        seed=int(seed),
    )


@dataclass(eq=False)
class DataSplit:
    """三段数据切分结果（架构类图 ``DataSplit``）。

    :param train: 训练段 ``Panel``。
    :param val: 验证段 ``Panel``。
    :param holdout: hold-out 段 ``Panel``。
    :param purge_gap: 段间 purge 间隔。
    :param embargo: 段间 embargo。
    :param holdout_seal: hold-out 指纹（写入封印台账的键）。
    :param sealed: 是否已成功封印。
    """

    train: Panel
    val: Panel
    holdout: Panel
    purge_gap: int
    embargo: int
    holdout_seal: str
    sealed: bool = False
    indices: SplitIndices | None = field(default=None, repr=False)

    def is_sealed(self) -> bool:
        """hold-out 是否已被一次性封印。"""
        return bool(self.sealed)


class HoldoutSealRegistry:
    """hold-out 一次性封印台账（JSON 落盘，默认 ``artifacts/holdout_seals.json``）。

    :param path: 台账文件路径；``None`` → :data:`DEFAULT_SEAL_PATH`。
        ``":memory:"`` → 纯内存台账（供测试隔离，不落盘）。
    """

    IN_MEMORY = ":memory:"

    def __init__(self, path: str | Path | None = None) -> None:
        self._memory = path is not None and str(path) == self.IN_MEMORY
        self.path: Path | None = None if self._memory else (
            Path(path) if path is not None else DEFAULT_SEAL_PATH
        )
        self._seals: dict[str, dict[str, Any]] = {}

    # ── 读 ──────────────────────────────────────────────────────────────────

    def _read(self) -> dict[str, dict[str, Any]]:
        if self._memory:
            return self._seals
        assert self.path is not None
        if not self.path.is_file():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DataError(f"封印台账损坏: {self.path}（{exc}）") from exc
        seals = data.get("seals", {})
        if not isinstance(seals, dict):
            raise DataError(f"封印台账 schema 非法: {self.path}")
        return {str(k): dict(v) for k, v in seals.items()}

    def is_sealed(self, fingerprint: str) -> bool:
        """判断某 hold-out 指纹是否已被使用过。"""
        return fingerprint in self._read()

    def list_seals(self) -> list[str]:
        """返回已封印的指纹列表（字典序）。"""
        return sorted(self._read())

    # ── 写 ──────────────────────────────────────────────────────────────────

    def seal(self, fingerprint: str, *, meta: dict[str, Any] | None = None) -> None:
        """封印 hold-out 指纹；若已存在则抛 :class:`HoldoutSealedError`。"""
        seals = self._read()
        if fingerprint in seals:
            raise HoldoutSealedError(
                "该 hold-out 指纹已被使用过（一次性封印），拒绝二次 peek",
                context={"fingerprint": fingerprint, "first": seals[fingerprint]},
            )
        entry: dict[str, Any] = {"seq": len(seals) + 1, "meta": dict(meta) if meta else {}}
        seals[fingerprint] = entry
        self._write(seals)

    def _write(self, seals: dict[str, dict[str, Any]]) -> None:
        if self._memory:
            self._seals = seals
            return
        assert self.path is not None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": _LEDGER_VERSION, "seals": seals}
        tmp = self.path.parent / (self.path.name + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "
", encoding="utf-8")
        os.replace(tmp, self.path)


def make_split(
    panel: Panel,
    split_config: SplitConfig,
    *,
    registry: HoldoutSealRegistry,
    seed: int,
    horizon: int = 2,
    seal: bool = True,
) -> DataSplit:
    """对 ``panel`` 做三段切分，并按需封印 hold-out。

    :param panel: 完整数据面板。
    :param split_config: 切分配置（比例 / purge / embargo / 折数）。
    :param registry: hold-out 封印台账。
    :param seed: 随机种子（参与切分指纹）。
    :param horizon: 前视期（仅用于日志/元信息；purge/embargo 由 ``split_config`` 给定）。
    :param seal: 是否封印 hold-out（``False`` 用于「仅检查可复现」的场景）。
    :raises HoldoutSealedError: hold-out 指纹已存在于台账。
    """
    ratios = (split_config.train_ratio, split_config.val_ratio, split_config.holdout_ratio)
    idx = three_way_split(
        panel.n_bars,
        ratios=ratios,
        purge_gap=split_config.purge_gap,
        embargo=split_config.embargo,
        seed=seed,
    )
    ranges = idx.ranges()
    train = panel.slice_view(*ranges["train"])
    val = panel.slice_view(*ranges["val"])
    holdout = panel.slice_view(*ranges["holdout"])

    split_fp = split_fingerprint(
        panel.fingerprint,
        ratios=ratios,
        purge_gap=split_config.purge_gap,
        embargo=split_config.embargo,
        seed=seed,
    )
    holdout_fp = holdout_fingerprint(split_fp, holdout.fingerprint)

    if seal:
        registry.seal(
            holdout_fp,
            meta={
                "panel_fingerprint": panel.fingerprint,
                "split_fingerprint": split_fp,
                "n_bars": panel.n_bars,
                "horizon": int(horizon),
                "seed": int(seed),
            },
        )

    return DataSplit(
        train=train,
        val=val,
        holdout=holdout,
        purge_gap=int(split_config.purge_gap),
        embargo=int(split_config.embargo),
        holdout_seal=holdout_fp,
        sealed=bool(seal),
        indices=idx,
    )
