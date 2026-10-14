# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for full license text.

"""WebUI 本地实时通道测试（路线图 R5.1）：四个 realtime 端点的 ``source=local``
返回**真实本地状态**，替代此前的「示值」假数据桩。

断言要点（诚实口径）：
* K 线：返回本地缓存 parquet 的**真实收盘价**（不再是 0.0 假蜡烛）；
* 报价：最后一根收盘价，且显式 ``source="cached_last_close"``（绝不伪造 0 报价）；
* 信号：对 watchlist 条目回放信号引擎 → 真实仓位/方向；
* 状态：watchlist 条目 + 门禁 verdict + kill 文件；
* 降级：无本地数据 → ``ok=False`` 信封（**不 500**、不伪造空值）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from miaosuan.core.vocab import VOCAB_VERSION
from miaosuan.ir.codec import write_spec
from miaosuan.ir.schema import Evidence, FactorPayload, Semantics, StrategySpec
from miaosuan.runtime import observer
from miaosuan.runtime.watchlist import WatchEntry, WatchlistStore
from miaosuan.search.rpn import random_feasible

# 复用 test_webui.py 的零依赖 ASGI 客户端（独立拷贝，避免跨测试文件 import）。
from test_webui import _call  # type: ignore[import-not-found]


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    """隔离产物目录 + 桩掉子进程，返回全新的 app（与 test_webui 同款）。"""
    import subprocess

    from miaosuan.webui import server

    artifacts = tmp_path / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(server, "_ARTIFACTS", artifacts)
    monkeypatch.setattr(server.subprocess, "Popen", _FakeProc)
    server._jobs._job = None
    created = server.create_app()
    yield created
    server._jobs._job = None


class _FakeProc:
    """``subprocess.Popen`` 替身：不真跑挖掘，但让 pump 线程正常收尾。"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs
        self.stdout: list[str] = []
        self.returncode = 0

    def wait(self, timeout: float | None = None) -> int:
        return self.returncode


def _make_spec() -> StrategySpec:
    rng = np.random.default_rng(42)
    tokens = tuple(int(t) for t in random_feasible(rng, 8))
    return StrategySpec(
        name="local_realtime_smoke",
        payload=FactorPayload(tokens=tokens, vocab_version=VOCAB_VERSION),
        semantics=Semantics(timeframe="M30"),  # 默认 warmup=499：历史足够才真正计算特征
        evidence=Evidence(gate_verdict="DEPLOYABLE"),
    )


def _write_kline(kline_dir: Path, n_bars: int = 520) -> None:
    """写确定性 M30 蜡烛（close 单调递增，价格带两位小数）。"""
    kline_dir.mkdir(parents=True, exist_ok=True)
    start = 1_800_000_000
    rows = []
    price = 4000.0
    for i in range(n_bars):
        price += 0.5
        rows.append(
            {
                "time": start + i * 1800,
                "open": round(price - 0.1, 2),
                "high": round(price + 0.2, 2),
                "low": round(price - 0.3, 2),
                "close": round(price, 2),
                "tick_volume": 100 + i,
            }
        )
    df = pd.DataFrame(rows)
    df.to_parquet(kline_dir / "XAUUSD_M30.parquet", index=False)


@pytest.fixture
def local_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    """装配真实本地环境：缓存 K 线 + watchlist 条目 + 产物目录。"""
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    kline_dir = tmp_path / "kline"
    _write_kline(kline_dir)

    spec_path = artifacts / "best.json"
    write_spec(spec_path, _make_spec())
    WatchlistStore(artifacts / "watchlist.json").save(
        [
            WatchEntry(
                symbol="XAUUSD",
                spec_path=str(spec_path),
                profile="FOREX_XAUUSD",
                timeframe="M30",
            )
        ]
    )

    # 只读观测路径全部指向临时环境（不碰真实 artifacts / 行情目录）。
    monkeypatch.setattr(observer, "_KLINE_DIRS", (kline_dir,))
    monkeypatch.setattr("miaosuan.runtime.watchlist.artifacts_dir", lambda: str(artifacts))
    monkeypatch.setattr("miaosuan.runtime.state.artifacts_dir", lambda: str(artifacts))
    return {"artifacts": artifacts, "kline": kline_dir, "spec": spec_path}


# ── K 线：真实蜡烛（不再是假 0.0 蜡烛）─────────────────────────────────────


def test_local_candles_returns_real_klines(app: Any, local_env: dict[str, Any]) -> None:
    status, data = _call(app, "GET", "/api/realtime/candles?source=local&symbol=XAUUSD&timeframe=M30&count=5")
    assert status == 200
    assert data["ok"] is True
    assert data["backend"] == "local"
    candles = data["data"]
    assert len(candles) == 5
    closes = [c["close"] for c in candles]
    # 真实单调递增收盘价（≈4599..4601），绝非示值桩的 0.0。
    assert closes == sorted(closes) and closes[0] > 4000
    # 升序时间。
    times = [c["time"] for c in candles]
    assert times == sorted(times)


def test_local_candles_degrades_honestly_when_no_data(app: Any, tmp_path: Path) -> None:
    """指定不存在的品种 → ok=False 降级信封（绝不 500、不伪造）。"""
    from miaosuan.webui import server

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(observer, "_KLINE_DIRS", (tmp_path / "empty",))
    monkeypatch.setattr("miaosuan.runtime.watchlist.artifacts_dir", lambda: str(tmp_path))
    monkeypatch.setattr("miaosuan.runtime.state.artifacts_dir", lambda: str(tmp_path))
    try:
        status, data = _call(
            app, "GET", "/api/realtime/candles?source=local&symbol=NOPE&timeframe=H1&count=5"
        )
    finally:
        monkeypatch.undo()
    assert status == 200
    assert data["ok"] is False
    assert data["backend"] == "local"
    assert data["data"] is None


# ── 报价：真实收盘价 + 显式来源（绝不伪造 0）───────────────────────────────


def test_local_price_returns_cached_last_close(app: Any, local_env: dict[str, Any]) -> None:
    status, data = _call(app, "GET", "/api/realtime/price?source=local&symbol=XAUUSD")
    assert status == 200
    assert data["ok"] is True
    price = data["data"]
    assert price["bid"] == price["ask"] > 4000  # 真实末根收盘价，非 0.0
    assert price["spread"] == 0.0
    assert price["source"] == "cached_last_close"
    assert price["symbol"] == "XAUUSD"


# ── 信号：真实仓位/方向（不是假 0.0 信号）──────────────────────────────────


def test_local_signals_reports_real_position(app: Any, local_env: dict[str, Any]) -> None:
    status, data = _call(app, "GET", "/api/realtime/signals?source=local")
    assert status == 200
    assert data["ok"] is True
    rows = data["data"]
    assert len(rows) == 1
    row = rows[0]
    assert row["strategy"] == "local_realtime_smoke"
    assert row["symbol"] == "XAUUSD"
    assert row["status"] == "signal"  # 引擎真实跑过，而不是 unavailable
    assert isinstance(row["position"], float)
    assert row["signal"] in ("多", "空", "观望")


# ── 引擎状态：watchlist + 门禁 + kill（真实只读）───────────────────────────


def test_local_status_reports_watchlist_and_verdict(
    app: Any, local_env: dict[str, Any]
) -> None:
    status, data = _call(app, "GET", "/api/realtime/status?source=local")
    assert status == 200
    assert data["ok"] is True
    st = data["data"]
    assert st["status"] == "local_runtime"
    assert len(st["watchlist"]) == 1
    item = st["watchlist"][0]
    assert item["symbol"] == "XAUUSD"
    assert item["gate_verdict"] == "DEPLOYABLE"
    assert item["spec_exists"] is True
    assert "kill_pending" in st and st["kill_pending"] is False


def test_local_status_empty_watchlist_ok(app: Any, tmp_path: Path) -> None:
    """空 watchlist → 仍 ok=True（本地通道可用，只是没有条目）。"""
    from miaosuan.webui import server

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(observer, "_KLINE_DIRS", (tmp_path / "k",))
    monkeypatch.setattr("miaosuan.runtime.watchlist.artifacts_dir", lambda: str(tmp_path))
    monkeypatch.setattr("miaosuan.runtime.state.artifacts_dir", lambda: str(tmp_path))
    try:
        status, data = _call(app, "GET", "/api/realtime/status?source=local")
    finally:
        monkeypatch.undo()
    assert status == 200
    assert data["ok"] is True
    assert data["data"]["watchlist"] == []


# ── 神机通道不受影响（回归防线）────────────────────────────────────────────


def test_shenji_channel_still_default(app: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """不传 source → 默认神机通道：走 _realtime_client 只读代理。"""
    from miaosuan.webui import server

    seen: dict[str, Any] = {}

    class _FakeClient:
        base_url = "http://127.0.0.1:1783"

        def engine_status(self) -> dict[str, Any]:
            seen["called"] = True
            return {"status": "running"}

    monkeypatch.setattr(server, "_realtime_client", lambda: _FakeClient())
    status, data = _call(app, "GET", "/api/realtime/status")
    assert status == 200
    assert data["ok"] is True
    assert data["backend"] == "http://127.0.0.1:1783"
    assert seen.get("called") is True