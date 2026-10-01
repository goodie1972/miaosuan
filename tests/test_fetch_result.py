# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for the full license text.

"""``fetch()`` 结构化返回（``FetchResult``）的契约测试——全程离线。

背景（2026-09 重构）：旧 ``fetch()`` 直接返回 ``Path``，把三种本质不同的
结局压成同一个路径值：

* 拉到新数据（真·成功）；
* 源正常响应、确认无新 bar（数据已最新，不是失败）；
* 所有源全挂、静默回退旧缓存（**假成功**——用户以为拿到新数据）。

第三种是项目系统性弱点「静默失败」的实例。本测试守护新契约：

* :class:`FetchStatus` 三态必须可区分；
* ``STALE_FALLBACK`` 必须携带各源失败原因（``errors``）；
* 部分源失败但整体成功 → ``warnings``（非致命）；
* 无缓存且全失败 → 仍然抛 ``DataError`` 且 ``context["failures"]`` 逐条在；
* ``path`` 实现 ``__fspath__``，旧调用方式（直接喂给 pandas）不破坏。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from miaosuan.data.acquisition import (
    DataAcquisition,
    DataSource,
    FetchResult,
    FetchStatus,
)
from miaosuan.errors import DataError

# ── 测试工具：可编排的假来源 ────────────────────────────────────────────────


def _df(n: int = 5, start: int = 1_700_000_000) -> pd.DataFrame:
    """构造最小合法 OHLCV（含 tick_volume，满足 _REQUIRED_COLUMNS）。"""
    return pd.DataFrame(
        {
            "time": [start + i * 3600 for i in range(n)],
            "open": [1.0] * n,
            "high": [2.0] * n,
            "low": [0.5] * n,
            "close": [1.5] * n,
            "volume": [10] * n,
            "tick_volume": [10] * n,
        }
    )


class ScriptedSource(DataSource):
    """按脚本应答的假来源：``full`` / ``incremental`` 各为数据或异常。

    值为 ``Exception`` 实例 → 抛出；值为 ``pd.DataFrame`` → 返回；
    ``"empty"`` → 返回空 DataFrame（源正常响应但无数据）。
    """

    def __init__(
        self,
        *,
        stype: str,
        full=None,
        incremental=None,
        label: str = "",
    ) -> None:
        self._stype = stype
        self._full = full
        self._inc = incremental
        self._label = label or stype

    @staticmethod
    def _answer(value) -> pd.DataFrame:
        if isinstance(value, Exception):
            raise value
        # 注意：必须先 isinstance 再比对——直接 `df == "empty"` 返回 DataFrame，
        # 放进 if 会抛 "truth value of a DataFrame is ambiguous"。
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


@pytest.fixture()
def cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """把缓存目录隔离到 tmp_path（同一目录 = 同一缓存文件）。"""
    monkeypatch.setenv("MIAOSUAN_DATA_CACHE_DIR", str(tmp_path))
    return tmp_path


def _acq(*sources: DataSource) -> DataAcquisition:
    return DataAcquisition(sources=list(sources))


# ── 1) UPDATED：拉到新数据 ──────────────────────────────────────────────────


def test_full_fetch_success_is_updated(cache_dir: Path) -> None:
    """首跑（无缓存）全量成功 → status=UPDATED，数据落盘。"""
    src = ScriptedSource(stype="Dukascopy", full=_df(), incremental=_df())
    res = _acq(src).fetch("XAUUSD", "H1")

    assert isinstance(res, FetchResult)
    assert res.status is FetchStatus.UPDATED
    assert res.source_used == "Dukascopy"
    assert res.errors == [] and res.warnings == []
    assert not res.stale
    assert res.path.is_file()
    assert len(pd.read_parquet(res.path)) == 5


def test_incremental_merge_is_updated(cache_dir: Path) -> None:
    """已有缓存 + 增量返回新 bar → UPDATED（合并去重后写回）。"""
    src = ScriptedSource(stype="Dukascopy", full=_df(n=5), incremental=_df(n=5, start=1_700_000_000 + 5 * 3600))
    acq = _acq(src)
    first = acq.fetch("XAUUSD", "H1")          # 首跑落 5 根
    assert first.status is FetchStatus.UPDATED

    second = acq.fetch("XAUUSD", "H1")          # 增量再 +5 根
    assert second.status is FetchStatus.UPDATED
    assert len(pd.read_parquet(second.path)) == 10


# ── 2) UP_TO_DATE：源确认无新 bar（**不是失败**） ──────────────────────────


def test_incremental_no_new_bar_is_up_to_date(cache_dir: Path) -> None:
    """缓存已是最新（增量源正常响应但为空）→ UP_TO_DATE + warnings=[]。"""
    src = ScriptedSource(stype="Dukascopy", full=_df(), incremental="empty")
    acq = _acq(src)
    acq.fetch("XAUUSD", "H1")                   # 建缓存

    res = acq.fetch("XAUUSD", "H1")
    assert res.status is FetchStatus.UP_TO_DATE
    assert res.source_used == "Dukascopy"
    assert res.errors == [] and res.warnings == []
    assert not res.stale
    assert res.path.is_file()                   # 旧缓存原样可用


# ── 3) STALE_FALLBACK：全失败 + 有旧缓存（旧版会静默返回，新版必须标注） ──


def test_all_sources_fail_with_cache_is_stale_fallback(cache_dir: Path) -> None:
    """所有源失败、本地有旧缓存 → STALE_FALLBACK + errors 逐条携带原因。"""
    good = ScriptedSource(stype="Dukascopy", full=_df(), incremental=_df())
    acq = _acq(good)
    acq.fetch("XAUUSD", "H1")                   # 先建一份「旧缓存」

    dead = ScriptedSource(
        stype="Dukascopy",
        full=DataError("网络不通"),
        incremental=DataError("网络不通"),
        label="Dukascopy(dead)",
    )
    res = _acq(dead).fetch("XAUUSD", "H1")

    assert res.status is FetchStatus.STALE_FALLBACK
    assert res.stale is True
    assert res.errors, "STALE_FALLBACK 必须携带失败原因，否则回到静默失败"
    assert any("网络不通" in e for e in res.errors)
    assert res.source_used == ""
    assert res.path.is_file()                   # 文件还在，但已如实标注过期


def test_stale_fallback_reports_every_failing_source(cache_dir: Path) -> None:
    """两个源都挂 → errors 里两个源的失败原因都在（逐条透传，不压缩成一句）。"""
    good = ScriptedSource(stype="Dukascopy", full=_df(), incremental=_df())
    _acq(good).fetch("XAUUSD", "H1")

    both_dead = _acq(
        ScriptedSource(stype="Dukascopy", full=DataError("boom-A"),
                       incremental=DataError("boom-A"), label="FakeA"),
        ScriptedSource(stype="TradingView", full=DataError("boom-B"),
                       incremental=DataError("boom-B"), label="FakeB"),
    )
    res = both_dead.fetch("XAUUSD", "H1")

    assert res.status is FetchStatus.STALE_FALLBACK
    assert any("boom-A" in e for e in res.errors)
    assert any("boom-B" in e for e in res.errors)


# ── 4) 无缓存 + 全失败 → 仍然抛 DataError（failures 在 context 里） ────────


def test_all_sources_fail_without_cache_raises_with_failures(cache_dir: Path) -> None:
    """无缓存可兜底 → DataError；消息与 context["failures"] 都逐条在。"""
    dead = _acq(
        ScriptedSource(stype="Dukascopy", full=DataError("boom-A"),
                       incremental=DataError("boom-A"), label="FakeA"),
        ScriptedSource(stype="TradingView", full=DataError("boom-B"),
                       incremental=DataError("boom-B"), label="FakeB"),
    )
    with pytest.raises(DataError) as excinfo:
        dead.fetch("XAUUSD", "H1")

    failures = excinfo.value.context.get("failures", [])
    assert any("boom-A" in f for f in failures)
    assert any("boom-B" in f for f in failures)
    # 错误消息本身也带原因（用户在弹窗里立刻看得见）
    assert "boom-A" in str(excinfo.value)
    assert "boom-B" in str(excinfo.value)


# ── 5) 部分源失败但整体成功 → warnings（非致命，不挡住成功） ───────────────


def test_partial_failure_success_has_warnings_not_errors(cache_dir: Path) -> None:
    """源 A 挂、源 B 成功 → UPDATED + warnings（A 的失败原因），errors=[]。"""
    res = _acq(
        ScriptedSource(stype="TradingView", full=DataError("tv 挂了"),
                       incremental=DataError("tv 挂了"), label="TradingView(dead)"),
        ScriptedSource(stype="Dukascopy", full=_df(), incremental=_df()),
    ).fetch("XAUUSD", "H1")

    assert res.status is FetchStatus.UPDATED
    assert res.source_used == "Dukascopy"
    assert res.errors == []
    assert any("tv 挂了" in w for w in res.warnings)


# ── 6) 兼容性：path 可直接喂给旧调用方 ─────────────────────────────────────


def test_result_path_is_pathlike(cache_dir: Path) -> None:
    """``__fspath__`` 使 pd.read_parquet(res) 直接可用（旧调用方式不破坏）。"""
    src = ScriptedSource(stype="Dukascopy", full=_df(), incremental=_df())
    res = _acq(src).fetch("XAUUSD", "H1")

    # 旧写法 pd.read_parquet(path)——现在传 FetchResult 本身
    back = pd.read_parquet(res)
    assert len(back) == 5
    assert Path(res) == res.path                # Path(result) 也接受
    assert str(res.path) in str(res.to_dict()["path"])


def test_to_dict_is_json_friendly(cache_dir: Path) -> None:
    """to_dict 输出 JSON 友好字段（WebUI 透传给前端用）。"""
    import json

    src = ScriptedSource(stype="Dukascopy", full=_df(), incremental=_df())
    res = _acq(src).fetch("XAUUSD", "H1")

    d = res.to_dict()
    assert d["status"] == "updated"
    assert d["stale"] is False
    assert d["source_used"] == "Dukascopy"
    json.dumps(d)  # 不抛 = 纯 JSON 类型


# ── 7) 显式 source → 文件名带 source 后缀（防不同源共用一份缓存） ──────────


def test_explicit_source_separates_cache_file(cache_dir: Path) -> None:
    """指定 source=Dukascopy → 缓存文件名含 _Dukascopy（与自动模式不互踩）。"""
    src = ScriptedSource(stype="Dukascopy", full=_df(), incremental=_df())
    res = _acq(src).fetch("XAUUSD", "D1", source="Dukascopy")

    assert "Dukascopy" in res.path.name
    assert (cache_dir / "XAUUSD_D1_Dukascopy.parquet").is_file()

    # 自动模式（source=None）保持原名，历史兼容
    res2 = _acq(src).fetch("BTCUSDT", "H1")
    assert "Dukascopy" not in res2.path.name
    assert (cache_dir / "BTCUSDT_H1.parquet").is_file()
