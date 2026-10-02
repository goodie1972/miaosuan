"""tools/duk_fetch_app.py 的单元测试（分批拉取 + 重叠验证 + 断点续传 v2）。

全部通过 monkeypatch `DukascopyFetcher.http_get` 返回合成数据，
禁止发起真实网络请求（`is_available` 一并打桩，避免 TCP 探测）。
"""

from __future__ import annotations

import importlib.util
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

# ── 按路径加载 tools/duk_fetch_app.py（自包含脚本，非包模块） ──
_APP_PATH = Path(__file__).resolve().parents[1] / "tools" / "duk_fetch_app.py"
_spec = importlib.util.spec_from_file_location("duk_fetch_app", _APP_PATH)
assert _spec is not None and _spec.loader is not None
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


# ══════════════════════════════════════════════════════════════════
# 合成数据
# ══════════════════════════════════════════════════════════════════

BARS = 8            # 每个工作日 8 根 M30 bar（00:00 ~ 03:30）
BAR_SEC = 1800      # M30 周期秒数
_TODAY = date(2026, 1, 21)   # 固定基准日期（保证批次窗口可预测）


def _day_price(d: date) -> float:
    """按日期确定的价格（整数部分保证 float 精确可比）"""
    return 1000.0 + (d.toordinal() - date(2006, 1, 1).toordinal()) % 500


def _make_payload(year: int, month: int, day: int) -> dict:
    """构造 Dukascopy 风格的 minute endpoint 返回（经 _parse_response 还原）"""
    base_ms = int(datetime(year, month, day, tzinfo=UTC).timestamp() * 1000)
    price = _day_price(date(year, month, day))
    n = BARS
    return {
        "timestamp": base_ms,
        "multiplier": 1.0,
        "shift": BAR_SEC * 1000,
        "times": [0] + [1] * (n - 1),
        "opens": [0] * n,
        "highs": [0] * n,
        "lows": [0] * n,
        "closes": [0] * n,
        "volumes": [1.0] * n,
        "open": price,
        "high": price,
        "low": price,
        "close": price,
    }


def _expected_rows(d: date) -> list[dict]:
    """与 _parse_response(_make_payload(...)) 等价的行（测试基准）"""
    base = int(datetime(d.year, d.month, d.day, tzinfo=UTC).timestamp())
    price = _day_price(d)
    return [
        {
            "time": base + i * BAR_SEC,
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": 1.0,
        }
        for i in range(BARS)
    ]


def _rows_for_days(days: list[date]) -> list[dict]:
    rows: list[dict] = []
    for d in days:
        rows.extend(_expected_rows(d))
    return rows


def _shift_rows(rows: list[dict], delta: float) -> list[dict]:
    """OHLC 整体平移 delta（用于构造 final 的差异值）"""
    out = []
    for r in rows:
        r2 = dict(r)
        for k in ("open", "high", "low", "close"):
            r2[k] = r2[k] + delta
        out.append(r2)
    return out


def _weekdays(start: date, end: date) -> list[date]:
    """[start, end) 内的工作日"""
    out: list[date] = []
    d = start
    while d < end:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


# ══════════════════════════════════════════════════════════════════
# 网络打桩
# ══════════════════════════════════════════════════════════════════

def _parse_url(url: str) -> tuple:
    parts = url.split("/")
    if "/minute/" in url:
        y, m, d = (int(x) for x in parts[-3:])
        return ("minute", y, m, d)
    if "/hour/" in url:
        y, m = (int(x) for x in parts[-2:])
        return ("hour", y, m, None)
    return ("day", int(parts[-1]), None, None)


def _url_label(url: str) -> str:
    kind, y, m, d = _parse_url(url)
    if kind == "minute":
        return f"{y}-{m:02d}-{d:02d}"
    if kind == "hour":
        return f"{y}-{m:02d}"
    return str(y)


class _FakeNet:
    """记录 http_get 调用，并可按 label 指定永久失败日期"""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.fail_labels: set[str] = set()

    def labels(self) -> set[str]:
        return {_url_label(u) for u in self.calls}


@pytest.fixture
def fake_net(monkeypatch):
    net = _FakeNet()

    def fake_http_get(self, url: str):
        net.calls.append(url)
        label = _url_label(url)
        if label in net.fail_labels:
            raise mod.DataError(f"simulated failure: {label}")
        kind, y, m, d = _parse_url(url)
        if kind != "minute":
            raise AssertionError(f"测试不应触发非 minute endpoint: {url}")
        return _make_payload(y, m, d)

    monkeypatch.setattr(mod.DukascopyFetcher, "http_get", fake_http_get)
    monkeypatch.setattr(mod.DukascopyFetcher, "is_available", lambda self: True)
    return net


# ══════════════════════════════════════════════════════════════════
# 测试工具
# ══════════════════════════════════════════════════════════════════

def _run(tmp_path, today, *, max_batches=1, resume=True, **kwargs):
    """调用 run_fetch（测试默认: XAUUSD / M30 / 小保存间隔 / 少日志）"""
    return mod.run_fetch(
        symbol="XAUUSD",
        timeframes=["M30"],
        cache_dir=tmp_path / "cache",
        checkpoint_path=tmp_path / "cp.json",
        log_file=tmp_path / "log.txt",
        proxy=None,
        save_every=kwargs.pop("save_every", 5),
        progress_every=kwargs.pop("progress_every", 1000),
        cp_every=kwargs.pop("cp_every", 1000),
        resume=resume,
        today=today,
        max_batches=max_batches,
        **kwargs,
    )


def _cache(tmp_path) -> Path:
    return tmp_path / "cache"


def _final_path(tmp_path) -> Path:
    return mod.final_parquet_path(_cache(tmp_path), "XAUUSD", "M30")


def _stage_path(tmp_path) -> Path:
    return mod.stage_parquet_path(_cache(tmp_path), "XAUUSD", "M30")


def _write_final(tmp_path, rows: list[dict]) -> Path:
    path = _final_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mod._finalize(pd.DataFrame(rows)).to_parquet(path, index=False)
    return path


def _read_final(tmp_path) -> pd.DataFrame:
    return pd.read_parquet(_final_path(tmp_path))


def _read_cp(tmp_path) -> dict:
    return json.loads((tmp_path / "cp.json").read_text(encoding="utf-8"))


def _write_cp(tmp_path, cp: dict) -> None:
    (tmp_path / "cp.json").write_text(
        json.dumps(cp, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _log_text(tmp_path) -> str:
    return (tmp_path / "log.txt").read_text(encoding="utf-8")


def _make_cp(
    today: date,
    *,
    batch_index: int = 0,
    batch_status: str = "pending",
    cursor: str | None = None,
) -> dict:
    """构造 v2 断点: 前 batch_index 个批次已 merged，目标批为 batch_status"""
    batches = mod.build_batches(today)
    entry = mod.build_tf_entry(batches)
    entry["status"] = "in_progress"
    entry["batch_index"] = batch_index
    for i in range(batch_index):
        entry["batches"][i]["status"] = "merged"
    target = entry["batches"][batch_index]
    target["status"] = batch_status
    if cursor is not None:
        target["cursor"] = cursor
    return {"version": 2, "overlap_days": 15, "symbol": "XAUUSD", "M30": entry}


# ══════════════════════════════════════════════════════════════════
# 1) 批次窗口与 15 天重叠正确性（含 2006-10-05 尾批）
# ══════════════════════════════════════════════════════════════════

def test_build_batches_windows_and_overlap():
    today = date(2026, 7, 9)
    batches = mod.build_batches(today, overlap_days=15)

    # batch0 = [当年-01-01, 今天+1天) —— 右开边界包含今天
    assert batches[0]["start"] == date(2026, 1, 1)
    assert batches[0]["end"] == date(2026, 7, 10)

    # 共 2026 → 2006 共 21 批
    assert len(batches) == 21

    # 下一批: [去年-01-01, 今年-01-16) —— 伸进已拉取区间 15 天
    assert batches[1]["start"] == date(2025, 1, 1)
    assert batches[1]["end"] == date(2026, 1, 16)
    assert batches[2]["end"] == date(2025, 1, 16)

    # 尾批: 2006-10-05 起
    last = batches[-1]
    assert last["start"] == date(2006, 10, 5)
    assert last["end"] == date(2007, 1, 16)

    # 每个老批次的 end == 年轻相邻批 start + overlap_days，且窗口合法
    for younger, older in zip(batches, batches[1:], strict=False):
        assert older["end"] == younger["start"] + timedelta(days=15)
        assert older["start"] < older["end"]
        assert older["start"] < younger["start"]


def test_build_batches_custom_overlap_days():
    batches = mod.build_batches(date(2026, 7, 9), overlap_days=7)
    assert batches[1]["end"] == date(2026, 1, 8)   # 年轻批 start + 7 天
    assert batches[-1]["start"] == date(2006, 10, 5)
    assert batches[-1]["end"] == date(2007, 1, 8)


# ══════════════════════════════════════════════════════════════════
# 2) 首次拉取: 只抓 batch0 窗口工作日，验证通过后合并
# ══════════════════════════════════════════════════════════════════

def test_fresh_run_fetches_only_batch0_window_and_merges(tmp_path, fake_net):
    _run(tmp_path, _TODAY, max_batches=1)

    expected_days = _weekdays(date(2026, 1, 1), date(2026, 1, 22))
    # 只抓 batch0 窗口内的工作日；无周末、无窗口外日期
    assert fake_net.labels() == {d.isoformat() for d in expected_days}

    final = _read_final(tmp_path)
    assert len(final) == len(expected_days) * BARS
    assert final["time"].is_unique
    assert final["time"].is_monotonic_increasing

    # 首次拉取（final 为空）跳过对比后合并；stage 已删除
    assert not _stage_path(tmp_path).exists()
    cp = _read_cp(tmp_path)
    assert cp["version"] == 2
    assert cp["symbol"] == "XAUUSD"
    b0 = cp["M30"]["batches"][0]
    assert b0["status"] == "merged"
    assert cp["M30"]["batch_index"] == 1
    assert cp["M30"]["status"] != "completed"
    assert "final 尚无数据" in _log_text(tmp_path)


# ══════════════════════════════════════════════════════════════════
# 3) 验证通过 → 合并正确（重叠区保留 final 值、重叠外追加、无重复、升序）
# ══════════════════════════════════════════════════════════════════

def test_batch1_verify_pass_merge_final_wins_and_appends(tmp_path, fake_net):
    # final: batch0 范围（2026-01-01 ~ 01-21）全量，值为真值 +0.005（容差内）
    final_rows = _shift_rows(_rows_for_days(_weekdays(date(2026, 1, 1), date(2026, 1, 22))), 0.005)
    _write_final(tmp_path, final_rows)
    # 断点: batch0 已 merged，本次处理 batch1 = [2025-01-01, 2026-01-16)
    _write_cp(tmp_path, _make_cp(_TODAY, batch_index=1, batch_status="pending"))

    _run(tmp_path, _TODAY, max_batches=1)

    # 抓取窗口严格 = batch1 内的工作日（含 15 天重叠区，不含 2026-01-16 之后）
    expected = {d.isoformat() for d in _weekdays(date(2025, 1, 1), date(2026, 1, 16))}
    assert fake_net.labels() == expected

    final = _read_final(tmp_path)
    assert final["time"].is_unique
    assert final["time"].is_monotonic_increasing

    # 重叠外追加: 2025 全年并入（stage 真值）
    assert len(final) == len(_weekdays(date(2025, 1, 1), date(2026, 1, 22))) * BARS
    d_2025 = date(2025, 1, 2)
    row_2025 = final[final["time"] == int(datetime(2025, 1, 2, tzinfo=UTC).timestamp())]
    assert len(row_2025) == 1
    assert row_2025.iloc[0]["close"] == pytest.approx(_day_price(d_2025))

    # 重叠区保留 final 值（+0.005 未被 stage 覆盖）
    t_0105 = int(datetime(2026, 1, 5, tzinfo=UTC).timestamp())
    row_0105 = final[final["time"] == t_0105]
    assert len(row_0105) == 1
    assert row_0105.iloc[0]["close"] == pytest.approx(_day_price(date(2026, 1, 5)) + 0.005)

    # stage 删除、状态推进
    assert not _stage_path(tmp_path).exists()
    cp = _read_cp(tmp_path)
    assert cp["M30"]["batches"][1]["status"] == "merged"
    assert cp["M30"]["batch_index"] == 2
    assert "重叠验证" in _log_text(tmp_path)


# ══════════════════════════════════════════════════════════════════
# 4) 验证失败 → 中止、不合并、stage 保留
# ══════════════════════════════════════════════════════════════════

def test_verify_fail_aborts_keeps_stage_and_final(tmp_path, fake_net):
    # final 全量但值偏移 0.5（远超容差 0.01）→ 必然验证失败
    final_rows = _shift_rows(_rows_for_days(_weekdays(date(2026, 1, 1), date(2026, 1, 22))), 0.5)
    _write_final(tmp_path, final_rows)
    before = _read_final(tmp_path)

    with pytest.raises(mod.VerificationError):
        _run(tmp_path, _TODAY, max_batches=1)

    # 不合并: final 原样保留
    after = _read_final(tmp_path)
    pd.testing.assert_frame_equal(before, after)
    # stage 保留
    assert _stage_path(tmp_path).exists()
    # 断点停在 fetched，等待排查后重新验证
    cp = _read_cp(tmp_path)
    assert cp["M30"]["batches"][0]["status"] == "fetched"
    # 日志含超容差原因与差异样本
    log = _log_text(tmp_path)
    assert "超容差" in log
    assert "差异样本" in log
    assert "已中止" in log


# ══════════════════════════════════════════════════════════════════
# 5) 重叠区补洞（final 缺一天，stage 补齐；其余 final 值保留）
# ══════════════════════════════════════════════════════════════════

def test_overlap_hole_filling_from_stage(tmp_path, fake_net):
    hole = date(2026, 1, 8)
    all_days = _weekdays(date(2026, 1, 1), date(2026, 1, 22))
    # final 缺 2026-01-08 整天，其余为真值 +0.005
    final_days = [d for d in all_days if d != hole]
    _write_final(tmp_path, _shift_rows(_rows_for_days(final_days), 0.005))
    _write_cp(tmp_path, _make_cp(_TODAY, batch_index=1, batch_status="pending"))

    _run(tmp_path, _TODAY, max_batches=1)

    final = _read_final(tmp_path)
    assert final["time"].is_unique
    assert final["time"].is_monotonic_increasing
    assert len(final) == len(_weekdays(date(2025, 1, 1), date(2026, 1, 22))) * BARS

    # 洞被 stage 补齐（stage 真值）
    hole_rows = final[
        (final["time"] >= int(datetime(2026, 1, 8, tzinfo=UTC).timestamp()))
        & (final["time"] < int(datetime(2026, 1, 9, tzinfo=UTC).timestamp()))
    ]
    assert len(hole_rows) == BARS
    assert hole_rows.iloc[0]["close"] == pytest.approx(_day_price(hole))

    # 重叠区原有 final 值未被覆盖
    t_0105 = int(datetime(2026, 1, 5, tzinfo=UTC).timestamp())
    row_0105 = final[final["time"] == t_0105]
    assert len(row_0105) == 1
    assert row_0105.iloc[0]["close"] == pytest.approx(_day_price(date(2026, 1, 5)) + 0.005)

    cp = _read_cp(tmp_path)
    assert cp["M30"]["batches"][1]["status"] == "merged"


# ══════════════════════════════════════════════════════════════════
# 6) resume: fetching —— 从游标续拉，不重复抓取
# ══════════════════════════════════════════════════════════════════

def test_resume_fetching_continues_from_cursor(tmp_path, fake_net):
    cursor_day = date(2026, 1, 9)
    # stage 已有 cursor 之前的数据；断点 status=fetching
    done_days = _weekdays(date(2026, 1, 1), cursor_day + timedelta(days=1))
    mod.save_stage_parquet(_stage_path(tmp_path), _rows_for_days(done_days))
    _write_cp(
        tmp_path,
        _make_cp(_TODAY, batch_index=0, batch_status="fetching", cursor=cursor_day.isoformat()),
    )

    _run(tmp_path, _TODAY, max_batches=1)

    # 只抓 cursor 之后的工作日（不抓 ≤cursor、不抓周末）
    expected_todo = {d.isoformat() for d in _weekdays(cursor_day + timedelta(days=1), date(2026, 1, 22))}
    assert fake_net.labels() == expected_todo

    # 合并后 final 含全窗口（stage 续拉数据 + 新抓数据），无重复升序
    final = _read_final(tmp_path)
    all_days = _weekdays(date(2026, 1, 1), date(2026, 1, 22))
    assert len(final) == len(all_days) * BARS
    assert final["time"].is_unique
    assert final["time"].is_monotonic_increasing

    cp = _read_cp(tmp_path)
    b0 = cp["M30"]["batches"][0]
    assert b0["status"] == "merged"
    assert b0["days_ok"] == len(all_days)
    assert b0["cursor"] == all_days[-1].isoformat()
    assert not _stage_path(tmp_path).exists()


# ══════════════════════════════════════════════════════════════════
# 7) resume: fetched —— 先验证再合并（通过 / 失败两档）
# ══════════════════════════════════════════════════════════════════

def _setup_fetched_batch0(tmp_path, final_delta: float | None) -> None:
    """stage = 真值全量；final = 真值(±delta)，断点 status=fetched"""
    days = _weekdays(date(2026, 1, 1), date(2026, 1, 22))
    mod.save_stage_parquet(_stage_path(tmp_path), _rows_for_days(days))
    rows = _rows_for_days(days)
    if final_delta is not None:
        rows = _shift_rows(rows, final_delta)
    _write_final(tmp_path, rows)
    _write_cp(tmp_path, _make_cp(_TODAY, batch_index=0, batch_status="fetched"))


def test_resume_fetched_verifies_then_merges(tmp_path, fake_net):
    _setup_fetched_batch0(tmp_path, final_delta=0.005)   # 容差内 → 验证通过
    before = _read_final(tmp_path)

    _run(tmp_path, _TODAY, max_batches=1)

    cp = _read_cp(tmp_path)
    assert cp["M30"]["batches"][0]["status"] == "merged"
    assert not _stage_path(tmp_path).exists()
    # stage 全部与 final 重叠 → 合并不新增行，final 值保留
    after = _read_final(tmp_path)
    assert len(after) == len(before)
    assert after["time"].is_unique
    t0 = int(datetime(2026, 1, 5, tzinfo=UTC).timestamp())
    assert after[after["time"] == t0].iloc[0]["close"] == pytest.approx(
        _day_price(date(2026, 1, 5)) + 0.005
    )
    # resume 路径同样触发验证
    assert "重叠验证" in _log_text(tmp_path)


def test_resume_fetched_mismatch_aborts(tmp_path, fake_net):
    _setup_fetched_batch0(tmp_path, final_delta=0.5)     # 超容差 → 验证失败
    before = _read_final(tmp_path)

    with pytest.raises(mod.VerificationError):
        _run(tmp_path, _TODAY, max_batches=1)

    pd.testing.assert_frame_equal(before, _read_final(tmp_path))
    assert _stage_path(tmp_path).exists()
    cp = _read_cp(tmp_path)
    assert cp["M30"]["batches"][0]["status"] == "fetched"


# ══════════════════════════════════════════════════════════════════
# 8) resume: verified —— 仍须验证，不得跳过直接合并
# ══════════════════════════════════════════════════════════════════

def test_resume_verified_proceeds_to_merge(tmp_path, fake_net):
    _setup_fetched_batch0(tmp_path, final_delta=0.005)
    # 断点已标 verified（例如崩溃在 verified → merge 之间）
    cp = _make_cp(_TODAY, batch_index=0, batch_status="verified")
    _write_cp(tmp_path, cp)

    _run(tmp_path, _TODAY, max_batches=1)

    out = _read_cp(tmp_path)
    assert out["M30"]["batches"][0]["status"] == "merged"
    assert not _stage_path(tmp_path).exists()


def test_resume_verified_still_verifies_before_merge(tmp_path, fake_net):
    # 即便状态是 verified，再次验证不通过也绝不合并
    _setup_fetched_batch0(tmp_path, final_delta=0.5)
    _write_cp(tmp_path, _make_cp(_TODAY, batch_index=0, batch_status="verified"))
    before = _read_final(tmp_path)

    with pytest.raises(mod.VerificationError):
        _run(tmp_path, _TODAY, max_batches=1)

    pd.testing.assert_frame_equal(before, _read_final(tmp_path))
    assert _stage_path(tmp_path).exists()
    cp = _read_cp(tmp_path)
    assert cp["M30"]["batches"][0]["status"] != "merged"


# ══════════════════════════════════════════════════════════════════
# 9) 旧 checkpoint 迁移: 不炸、保留最终文件、从 batch0 重来
# ══════════════════════════════════════════════════════════════════

def test_old_checkpoint_migration(tmp_path, fake_net):
    old_cp = {
        "M30": {
            "status": "in_progress",
            "year": 2024,
            "month": 5,
            "day": 3,
            "days_done": 100,
            "days_failed": 2,
        }
    }
    (tmp_path / "cp.json").write_text(json.dumps(old_cp), encoding="utf-8")
    # 保留既有最终文件（只含一天数据）
    _write_final(tmp_path, _rows_for_days([date(2026, 1, 5)]))
    assert _final_path(tmp_path).exists()

    _run(tmp_path, _TODAY, max_batches=1)   # 不应崩溃

    log = _log_text(tmp_path)
    assert "旧格式" in log

    cp = _read_cp(tmp_path)
    assert cp["version"] == 2
    assert cp["symbol"] == "XAUUSD"
    assert "year" not in cp                      # 旧字段不复存在
    assert isinstance(cp["M30"]["batches"], list)
    assert cp["M30"]["batches"][0]["status"] == "merged"

    # 最终文件保留并与新数据合并（无重复、升序）
    final = _read_final(tmp_path)
    assert len(final) == len(_weekdays(date(2026, 1, 1), date(2026, 1, 22))) * BARS
    assert final["time"].is_unique
    assert final["time"].is_monotonic_increasing
    assert not _stage_path(tmp_path).exists()


# ══════════════════════════════════════════════════════════════════
# 10) 批末重试: 失败日最多 2 轮重试，仍失败记入报告
# ══════════════════════════════════════════════════════════════════

def test_failed_day_retried_twice_then_reported(tmp_path, fake_net):
    bad_day = "2026-01-07"
    fake_net.fail_labels.add(bad_day)

    _run(tmp_path, _TODAY, max_batches=1)

    # 初次 1 次 + 重试 2 轮 = 3 次调用
    bad_urls = [u for u in fake_net.calls if u.endswith("/2026/1/7")]
    assert len(bad_urls) == 3

    cp = _read_cp(tmp_path)
    b0 = cp["M30"]["batches"][0]
    assert b0["status"] == "merged"
    assert b0["failed_days"] == [bad_day]
    assert b0["days_failed"] == 1
    assert b0["days_ok"] == len(_weekdays(date(2026, 1, 1), date(2026, 1, 22))) - 1

    # final 中该日留洞（已报告，不做静默填充）
    final = _read_final(tmp_path)
    bad_ts = int(datetime(2026, 1, 7, tzinfo=UTC).timestamp())
    assert not ((final["time"] >= bad_ts) & (final["time"] < bad_ts + 86400)).any()

    log = _log_text(tmp_path)
    assert "重试第 1/2 轮" in log
    assert "重试第 2/2 轮" in log
    assert bad_day in log


# ══════════════════════════════════════════════════════════════════
# 11) 验证函数单元: 缺口检测 / final 为空跳过
# ══════════════════════════════════════════════════════════════════

def test_verify_overlap_detects_stage_gap():
    day = date(2026, 1, 5)
    full = _expected_rows(day)
    stage_rows = full[:3] + full[4:]          # 缺 01:30 一根
    final_df = mod._finalize(pd.DataFrame(full))
    stage_df = mod._finalize(pd.DataFrame(stage_rows))

    ok, report = mod.verify_batch_overlap(
        final_df, stage_df, "2026-01-05", "2026-01-06",
        interval_sec=BAR_SEC, tol_price=0.01, tol_mismatch_pct=0.5,
    )
    assert not ok
    assert report["gap_count"] == 1
    assert any("缺口" in r for r in report["reasons"])

    # 完全一致时通过
    ok2, report2 = mod.verify_batch_overlap(
        final_df, mod._finalize(pd.DataFrame(full)), "2026-01-05", "2026-01-06",
        interval_sec=BAR_SEC, tol_price=0.01, tol_mismatch_pct=0.5,
    )
    assert ok2
    assert report2["gap_count"] == 0
    assert report2["mismatch_pct"] == 0.0


def test_verify_overlap_skips_when_final_empty():
    day = date(2026, 1, 5)
    stage_df = mod._finalize(pd.DataFrame(_expected_rows(day)))
    empty_final = pd.DataFrame(columns=[*mod.COLUMNS])

    ok, report = mod.verify_batch_overlap(
        empty_final, stage_df, "2026-01-01", "2026-01-22",
        interval_sec=BAR_SEC,
    )
    assert ok
    assert report["note"] is not None

    # stage 重叠区为空 → 失败
    ok2, report2 = mod.verify_batch_overlap(
        empty_final, pd.DataFrame(columns=[*mod.COLUMNS]), "2026-01-01", "2026-01-22",
        interval_sec=BAR_SEC,
    )
    assert not ok2
    assert report2["reasons"]


# ══════════════════════════════════════════════════════════════════
# 12) CLI 参数: 默认值与新增参数
# ══════════════════════════════════════════════════════════════════

def test_cli_default_args_parse():
    args = mod.parse_args([])
    assert args.symbol == "XAUUSD"
    assert args.timeframes == ["M30", "M15", "M5"]
    assert args.cache_dir == "data/cache"
    assert args.checkpoint == "tmp/duk_checkpoint.json"
    assert args.log_file == "tmp/duk_fetch_log.txt"
    assert args.proxy is None
    assert args.no_resume is False
    assert args.save_every == 50
    assert args.progress_every == 10
    assert args.cp_every == 5
    # 新增参数默认值
    assert args.overlap_days == 15
    assert args.tol_price == 0.01
    assert args.tol_mismatch_pct == 0.5
    assert args.max_batches is None


def test_cli_new_args_parse():
    args = mod.parse_args(
        [
            "--overlap-days", "7",
            "--tol-price", "0.02",
            "--tol-mismatch-pct", "1.0",
            "--max-batches", "2",
            "--proxy", "http://127.0.0.1:10808",
        ]
    )
    assert args.overlap_days == 7
    assert args.tol_price == 0.02
    assert args.tol_mismatch_pct == 1.0
    assert args.max_batches == 2
    assert args.proxy == "http://127.0.0.1:10808"


# ══════════════════════════════════════════════════════════════════
# 13) R3: final 文件损坏 → 中止报错、不放行、不覆盖 final
# ══════════════════════════════════════════════════════════════════

def test_corrupt_final_aborts_and_never_overwritten(tmp_path, fake_net):
    path = _final_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    garbage = b"NOT-A-PARQUET-FILE-THIS-IS-CORRUPT" * 8
    path.write_bytes(garbage)

    with pytest.raises(mod.DataError) as exc_info:
        _run(tmp_path, _TODAY, max_batches=1)

    msg = str(exc_info.value)
    assert str(path) in msg          # 错误信息定位到具体文件
    assert "处理建议" in msg          # 给出可操作的修复步骤
    # 绝不覆盖 final（历史数据原样保留）
    assert path.read_bytes() == garbage
    # 绝不放行：未进入合并、stage 未产生
    assert not _stage_path(tmp_path).exists()
    # 预检阶段即中止，不浪费任何抓取
    assert fake_net.calls == []
    assert "已中止" in _log_text(tmp_path)


# ══════════════════════════════════════════════════════════════════
# 14) R2: 崩溃窗口（final 已含数据 + stage 已删 + 断点=verified）
#      → resume 直接推进，不死锁、不覆盖 final
# ══════════════════════════════════════════════════════════════════

def test_r2_verified_without_stage_advances_without_deadlock(tmp_path, fake_net):
    days = _weekdays(date(2026, 1, 1), date(2026, 1, 22))
    # final 已写入合并结果（write_final 成功），stage 已被删除，
    # 但断点仍停在 verified（崩溃在删 stage 与存断点之间）
    _write_final(tmp_path, _rows_for_days(days))
    assert not _stage_path(tmp_path).exists()
    _write_cp(tmp_path, _make_cp(_TODAY, batch_index=0, batch_status="verified"))
    before = _read_final(tmp_path)

    _run(tmp_path, _TODAY, max_batches=1)   # 不得抛 VerificationError 死锁

    # final 原样保留（不被覆盖、历史不丢）
    pd.testing.assert_frame_equal(before, _read_final(tmp_path))
    # 断点推进
    cp = _read_cp(tmp_path)
    assert cp["M30"]["batches"][0]["status"] == "merged"
    assert cp["M30"]["batch_index"] == 1
    assert not _stage_path(tmp_path).exists()
    # 推进已计入 max_batches → 不再发起抓取
    assert fake_net.calls == []
    assert "直接推进" in _log_text(tmp_path)


# ══════════════════════════════════════════════════════════════════
# 15) R1: 批内 checkpoint 携带游标 → 中途崩溃后续拉不整批重抓
# ══════════════════════════════════════════════════════════════════

class _SimCrash(BaseException):
    """模拟进程崩溃（BaseException，穿透 fetch 内部的 except Exception）"""


def _install_crash(monkeypatch, crash_after: int):
    """第 crash_after+1 次网络调用时抛 _SimCrash；返回解除打桩的函数"""
    state = {"n": 0, "armed": True}
    orig = mod.DukascopyFetcher.http_get

    def crashing(self, url: str):
        state["n"] += 1
        if state["armed"] and state["n"] > crash_after:
            raise _SimCrash(f"simulated crash after {crash_after} calls")
        return orig(self, url)

    monkeypatch.setattr(mod.DukascopyFetcher, "http_get", crashing)
    return lambda: state.update(armed=False)


def test_mid_batch_checkpoint_advances_cursor_for_resume(tmp_path, fake_net, monkeypatch):
    # save_every=1000 → 只有 cp_every=3 节奏触发批内 checkpoint（原缺陷场景:
    # 批内 checkpoint 内容空转、cursor 只在批末写入）
    disarm = _install_crash(monkeypatch, crash_after=6)

    with pytest.raises(_SimCrash):
        _run(tmp_path, _TODAY, max_batches=1, save_every=1000, cp_every=3)

    # 崩溃点断点: fetching + 游标已推进到第 6 个工作日 2026-01-08
    cp = _read_cp(tmp_path)
    b0 = cp["M30"]["batches"][0]
    assert b0["status"] == "fetching"
    assert cp["M30"]["batch_index"] == 0
    assert b0["cursor"] == "2026-01-08"
    assert b0["days_ok"] == 6
    assert b0["days_failed"] == 0
    assert b0["failed_days"] == []
    # stage 已落盘（游标绝不领先 stage）
    stage = pd.read_parquet(_stage_path(tmp_path))
    assert len(stage) == 6 * BARS

    # resume: 从游标续拉
    disarm()
    _run(tmp_path, _TODAY, max_batches=1, save_every=1000, cp_every=3)

    # 每个工作日全程只抓一次 —— 未整批重抓
    labels = [_url_label(u) for u in fake_net.calls]
    assert len(labels) == 15
    assert len(set(labels)) == 15
    assert set(labels) == {d.isoformat() for d in _weekdays(date(2026, 1, 1), date(2026, 1, 22))}

    final = _read_final(tmp_path)
    assert len(final) == 15 * BARS
    assert final["time"].is_unique
    cp = _read_cp(tmp_path)
    assert cp["M30"]["batches"][0]["status"] == "merged"
    assert not _stage_path(tmp_path).exists()
