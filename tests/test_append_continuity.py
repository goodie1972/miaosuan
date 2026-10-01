# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""``fetch(append_to=...)`` 追加模式的连续性校验契约测试——全程离线。

用户场景（2026-09-30）：在「本地缓存」列表选一个文件，把新拉的增量**接到
该文件末尾**。但「添加前要确认数据是准确的、能连续下去」——接错数据比没数据
更危险（周期错配、时间单位混用、大段缺口都会静默污染后续所有回测）。

本测试守护两层校验：

1. **文件名层**（不读盘）：目标文件名声明的品种/周期 vs 请求——错配即拒绝；
2. **数据层**（读盘后）：数据实际周期、时间戳单位、时间方向、缺口分级——
   任何一条不过即抛 ``DataError`` 且**文件不被修改**。

以及追加模式的行为边界：

* 只做增量，**不走全量覆盖**（全量会把目标文件整体替换掉）；
* 增量失败 → ``STALE_FALLBACK``，文件保持原样；
* 校验通过但无净新增 → ``UP_TO_DATE``（不写盘）。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from miaosuan.data.acquisition import (
    DataAcquisition,
    DataSource,
    FetchStatus,
)
from miaosuan.errors import DataError

# ── 测试工具 ────────────────────────────────────────────────────────────────


def _df(n: int = 10, start: int = 1_700_000_000, step: int = 3600) -> pd.DataFrame:
    """构造合法 OHLCV；step=3600 表示 H1。"""
    return pd.DataFrame(
        {
            "time": [start + i * step for i in range(n)],
            "open": [1.0] * n,
            "high": [2.0] * n,
            "low": [0.5] * n,
            "close": [1.5] * n,
            "volume": [10] * n,
            "tick_volume": [10] * n,
        }
    )


class ScriptedSource(DataSource):
    """按脚本应答的假来源（同 test_fetch_result，独立副本避免跨文件耦合）。"""

    def __init__(self, *, stype: str, full=None, incremental=None, label: str = "") -> None:
        self._stype = stype
        self._full = full
        self._inc = incremental
        self._label = label or stype

    @staticmethod
    def _answer(value) -> pd.DataFrame:
        if isinstance(value, Exception):
            raise value
        if isinstance(value, str) and value == "empty":
            return pd.DataFrame(columns=[
                "time", "open", "high", "low", "close", "volume", "tick_volume",
            ])
        return value

    def fetch_full(self, symbol: str, timeframe: str) -> pd.DataFrame:  # noqa: ARG002
        return self._answer(self._full)

    def fetch_incremental(
        self, symbol: str, timeframe: str, since_ts: int  # noqa: ARG002
    ) -> pd.DataFrame:
        return self._answer(self._inc)

    def is_available(self) -> bool:
        return True

    def describe(self) -> str:
        return self._label

    def source_type(self) -> str:
        return self._stype


def _acq(*sources: DataSource) -> DataAcquisition:
    return DataAcquisition(sources=list(sources))


def _make_target(tmp_path: Path, name: str, df: pd.DataFrame | None = None) -> Path:
    """在 tmp_path 落一个目标 parquet（默认 10 根 H1，末端 1_700_032_400）。"""
    p = tmp_path / name
    (df if df is not None else _df()).to_parquet(p, index=False)
    return p


@pytest.fixture()
def cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("MIAOSUAN_DATA_CACHE_DIR", str(tmp_path))
    return tmp_path


# ── 1) 文件名层校验：品种/周期错配即拒绝（读盘前） ─────────────────────────


def test_append_rejects_timeframe_mismatch_in_filename(cache_dir: Path) -> None:
    """目标文件名是 D1、请求 H1 → 拒绝，且**不读文件**（文件不存在也该报错配）。"""
    dead = ScriptedSource(stype="Dukascopy", full=_df(), incremental=_df())
    # 文件故意不存在：若先报「不存在」说明校验顺序错了（应该先查文件名）
    missing = cache_dir / "XAUUSD_D1.parquet"
    with pytest.raises(DataError) as excinfo:
        _acq(dead).fetch("XAUUSD", "H1", append_to=missing)
    # 文件名错配优先于「不存在」被报出来
    assert "文件名周期" in str(excinfo.value)


def test_append_rejects_symbol_mismatch_in_filename(cache_dir: Path) -> None:
    """目标是 EURUSD 文件、请求 XAUUSD → 拒绝。"""
    src = ScriptedSource(stype="Dukascopy", full=_df(), incremental=_df())
    target = _make_target(cache_dir, "EURUSD_H1.parquet")
    with pytest.raises(DataError) as excinfo:
        _acq(src).fetch("XAUUSD", "H1", append_to=target)
    assert "文件名品种" in str(excinfo.value)
    # 文件未被修改
    assert len(pd.read_parquet(target)) == 10


def test_append_accepts_matching_filename(cache_dir: Path) -> None:
    """文件名匹配（含 note 后缀如 XAUUSD_H1_Dukascopy）→ 放行到数据层。"""
    new_df = _df(n=5, start=1_700_000_000 + 10 * 3600)
    src = ScriptedSource(stype="Dukascopy", full=_df(), incremental=new_df)
    target = _make_target(cache_dir, "XAUUSD_H1_Dukascopy.parquet")
    res = _acq(src).fetch("XAUUSD", "H1", append_to=target)
    assert res.status is FetchStatus.UPDATED
    assert len(pd.read_parquet(target)) == 15


# ── 2) 数据层校验：周期 / 单位 / 方向 / 缺口 ────────────────────────────────


def test_append_rejects_data_period_mismatch(cache_dir: Path) -> None:
    """目标文件实际是 D1 数据（文件名却叫 H1）→ 数据层周期校验兜底拒绝。"""
    d1_df = _df(n=10, step=86400)  # 实际周期 D1
    target = _make_target(cache_dir, "XAUUSD_H1.parquet", d1_df)
    new_df = _df(n=5, start=int(d1_df["time"].max()) + 3600, step=3600)
    src = ScriptedSource(stype="Dukascopy", full=d1_df, incremental=new_df)

    with pytest.raises(DataError) as excinfo:
        _acq(src).fetch("XAUUSD", "H1", append_to=target)
    assert "周期" in str(excinfo.value)
    assert len(pd.read_parquet(target)) == 10  # 未写入


def test_append_rejects_timestamp_unit_mismatch(cache_dir: Path) -> None:
    """旧数据秒级、新数据毫秒级 → 拒绝（混单位合并会搅乱时间轴）。

    注意：毫秒数据的多根 bar 会先被**周期校验**拦下（间隔 3600000 vs 3600，
    必然差 1000 倍）——要单独触达单位校验，新数据必须只有 1 根 bar
    （中位间隔算不出 → 周期校验跳过 → 单位校验接手）。
    """
    target = _make_target(cache_dir, "XAUUSD_H1.parquet")  # 秒级 10 根
    one_ms_bar = _df(n=1, start=(1_700_000_000 + 10 * 3600) * 1000)
    src = ScriptedSource(stype="Dukascopy", full=_df(), incremental=one_ms_bar)

    with pytest.raises(DataError) as excinfo:
        _acq(src).fetch("XAUUSD", "H1", append_to=target)
    assert "时间戳单位" in str(excinfo.value)
    assert len(pd.read_parquet(target)) == 10  # 未写入


def test_append_rejects_no_time_extension(cache_dir: Path) -> None:
    """新数据全落在旧区间内（源没按 since 过滤）→ 拒绝，避免空转写盘。"""
    target = _make_target(cache_dir, "XAUUSD_H1.parquet")
    stale_new = _df(n=5)  # 时间范围完全在旧数据前半段
    src = ScriptedSource(stype="Dukascopy", full=_df(), incremental=stale_new)

    with pytest.raises(DataError) as excinfo:
        _acq(src).fetch("XAUUSD", "H1", append_to=target)
    assert "未延伸时间轴" in str(excinfo.value)


def test_append_rejects_huge_gap(cache_dir: Path) -> None:
    """新旧缺口 45 天（> max(30天, 3×H1)）→ 拒绝：数据接不下去。"""
    target = _make_target(cache_dir, "XAUUSD_H1.parquet")
    gap_start = 1_700_000_000 + 10 * 3600 + 45 * 86400  # 45 天后才接上
    new_df = _df(n=5, start=gap_start)
    src = ScriptedSource(stype="Dukascopy", full=_df(), incremental=new_df)

    with pytest.raises(DataError) as excinfo:
        _acq(src).fetch("XAUUSD", "H1", append_to=target)
    assert "缺口" in str(excinfo.value)
    assert len(pd.read_parquet(target)) == 10  # 文件原样


def test_append_warns_on_medium_gap_but_writes(cache_dir: Path) -> None:
    """中档缺口（10 天，> max(7天, 5×H1) 但 < 30 天）→ 通过但 warnings 提示。"""
    target = _make_target(cache_dir, "XAUUSD_H1.parquet")
    gap_start = 1_700_000_000 + 10 * 3600 + 10 * 86400  # 10 天后接上
    new_df = _df(n=5, start=gap_start)
    src = ScriptedSource(stype="Dukascopy", full=_df(), incremental=new_df)

    res = _acq(src).fetch("XAUUSD", "H1", append_to=target)
    assert res.status is FetchStatus.UPDATED
    assert any("缺口" in w for w in res.warnings), "中档缺口必须出现在 warnings"
    assert len(pd.read_parquet(target)) == 15


def test_append_small_gap_silent_success(cache_dir: Path) -> None:
    """正常衔接（缺口 = 1 根 H1）→ 无警告、直接 UPDATE。"""
    target = _make_target(cache_dir, "XAUUSD_H1.parquet")
    new_df = _df(n=5, start=1_700_000_000 + 11 * 3600)
    src = ScriptedSource(stype="Dukascopy", full=_df(), incremental=new_df)

    res = _acq(src).fetch("XAUUSD", "H1", append_to=target)
    assert res.status is FetchStatus.UPDATED
    assert res.warnings == []
    back = pd.read_parquet(target)
    assert len(back) == 15
    assert back["time"].is_monotonic_increasing
    assert not back["time"].duplicated().any()


# ── 3) 追加模式行为边界 ─────────────────────────────────────────────────────


def test_append_target_missing_raises(cache_dir: Path) -> None:
    """追加目标不存在 → DataError（追加语义要求目标已存在）。

    用**文件名匹配但不存在**的名字——若用 ``nope.parquet`` 会先被文件名校验
    拦下（品种 nope ≠ XAUUSD），测不到存在性检查这一层。
    """
    src = ScriptedSource(stype="Dukascopy", full=_df(), incremental=_df())
    with pytest.raises(DataError) as excinfo:
        _acq(src).fetch(
            "XAUUSD", "H1", append_to=cache_dir / "XAUUSD_H1_notthere.parquet"
        )
    assert "追加目标不存在" in str(excinfo.value)


def test_append_incremental_fail_returns_stale_not_full_overwrite(
    cache_dir: Path,
) -> None:
    """增量失败 → STALE_FALLBACK；**绝不走全量覆盖**（全量会抹掉目标文件）。"""
    target = _make_target(cache_dir, "XAUUSD_H1.parquet")
    # full 挂了也无所谓——追加模式压根不该走到全量路径
    dead = ScriptedSource(
        stype="Dukascopy",
        full=DataError("全量也不该被调用"),
        incremental=DataError("网络不通"),
    )
    res = _acq(dead).fetch("XAUUSD", "H1", append_to=target)

    assert res.status is FetchStatus.STALE_FALLBACK
    assert any("网络不通" in e for e in res.errors)
    # 文件原样，10 根未动
    assert len(pd.read_parquet(target)) == 10


def test_append_no_new_data_is_up_to_date(cache_dir: Path) -> None:
    """增量源正常响应但无新 bar → UP_TO_DATE，不写盘。"""
    target = _make_target(cache_dir, "XAUUSD_H1.parquet")
    src = ScriptedSource(stype="Dukascopy", full=_df(), incremental="empty")
    res = _acq(src).fetch("XAUUSD", "H1", append_to=target)
    assert res.status is FetchStatus.UP_TO_DATE
    assert len(pd.read_parquet(target)) == 10


def test_append_non_parquet_target_rejected(cache_dir: Path) -> None:
    """csv 目标 → 拒绝（追加模式仅支持 parquet）。"""
    csv_target = cache_dir / "XAUUSD_H1.csv"
    csv_target.write_text("time,close\n1700000000,1.5\n", encoding="utf-8")
    src = ScriptedSource(stype="Dukascopy", full=_df(), incremental=_df())
    with pytest.raises(DataError) as excinfo:
        _acq(src).fetch("XAUUSD", "H1", append_to=csv_target)
    assert "仅支持 parquet" in str(excinfo.value)


def test_append_corrupt_target_raises_readable(cache_dir: Path) -> None:
    """目标文件损坏 → DataError 带「文件可能损坏」提示（可行动的报错）。"""
    target = cache_dir / "XAUUSD_H1.parquet"
    target.write_bytes(b"not a parquet at all")
    src = ScriptedSource(stype="Dukascopy", full=_df(), incremental=_df())
    with pytest.raises(DataError) as excinfo:
        _acq(src).fetch("XAUUSD", "H1", append_to=target)
    assert "损坏" in str(excinfo.value)


def test_append_dedupes_overlap_without_inflation(cache_dir: Path) -> None:
    """源返回的增量含 2 根重叠 bar → 去重后不膨胀（时间轴仍严格递增）。"""
    target = _make_target(cache_dir, "XAUUSD_H1.parquet")  # 末端 1_700_032_400
    # 新数据：从旧末端前 2 根开始（重叠 2 根）+ 5 根新 bar
    overlap_start = 1_700_000_000 + 8 * 3600
    new_df = _df(n=7, start=overlap_start)
    src = ScriptedSource(stype="Dukascopy", full=_df(), incremental=new_df)

    res = _acq(src).fetch("XAUUSD", "H1", append_to=target)
    assert res.status is FetchStatus.UPDATED
    back = pd.read_parquet(target)
    assert len(back) == 15  # 10 + 5（2 根重叠被去重）
    assert back["time"].is_monotonic_increasing
    assert not back["time"].duplicated().any()
