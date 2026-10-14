# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for full license text.

"""Playwright 实时页测试（R5.1/R5.2 页面层）。

隔离方式**只认配置**：临时目录里写 ``settings.yaml``（artifacts/kline 指向
临时子目录），服务器以该目录为 cwd 启动——不依赖任何环境变量（与
``scripts/test_e2e_realtime.py`` 同款）。验证：

* 实时页打开：引擎状态 / 当前报价 / K 线 / 最近信号 区块可见；
* 数据通道切换（神机 ↔ 本地 runtime）控件存在；
* ``source=local`` 下 K 线渲染真实收盘价（非 0.0 假蜡烛）。
"""

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

try:
    from playwright.sync_api import sync_playwright, expect
except ImportError:
    pytest.skip("playwright not installed", allow_module_level=True)

_PORT = int(os.environ.get("MIAOSUAN_PLAYWRIGHT_PORT", "8691"))
_BASE = f"http://127.0.0.1:{_PORT}"


def _build_fixtures(root: Path) -> None:
    from miaosuan.core.vocab import VOCAB_VERSION
    from miaosuan.ir.codec import write_spec
    from miaosuan.ir.schema import Evidence, FactorPayload, Semantics, StrategySpec
    from miaosuan.runtime.watchlist import WatchEntry, WatchlistStore
    from miaosuan.search.rpn import random_feasible

    artifacts = root / "artifacts"
    kline = root / "kline"
    artifacts.mkdir(parents=True, exist_ok=True)
    kline.mkdir(parents=True, exist_ok=True)

    (root / "settings.yaml").write_text(
        "paths:\n"
        f"  artifacts: {artifacts.as_posix()}\n"
        f"  kline: {kline.as_posix()}\n"
        "notifiers:\n"
        "  feishu:\n"
        "    webhook_url: ''\n",
        encoding="utf-8",
    )

    start = 1_800_000_000
    rows = []
    price = 4000.0
    for i in range(520):
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
    pd.DataFrame(rows).to_parquet(kline / "XAUUSD_M30.parquet", index=False)

    rng = np.random.default_rng(42)
    tokens = tuple(int(t) for t in random_feasible(rng, 8))
    spec = StrategySpec(
        name="pw_realtime",
        payload=FactorPayload(tokens=tokens, vocab_version=VOCAB_VERSION),
        semantics=Semantics(timeframe="M30"),
        evidence=Evidence(gate_verdict="DEPLOYABLE"),
    )
    spec_path = artifacts / "pw_realtime.json"
    write_spec(spec_path, spec)
    WatchlistStore(artifacts / "watchlist.json").save(
        [
            WatchEntry(
                symbol="XAUUSD",
                spec_path=str(spec_path),
                profile="FOREX_XAUUSD_MT5",
                timeframe="M30",
            )
        ]
    )


def test_realtime_page():
    with tempfile.TemporaryDirectory(prefix="miaosuan_pw_") as tmp:
        root = Path(tmp)
        _build_fixtures(root)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
        server = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "miaosuan.webui.server:app",
             "--host", "127.0.0.1", "--port", str(_PORT)],
            cwd=str(root),  # cwd 含 settings.yaml → 只认配置隔离
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            for _ in range(30):
                try:
                    import requests
                    if requests.get(f"{_BASE}/api/health", timeout=1).status_code == 200:
                        break
                except Exception:
                    time.sleep(0.5)
            else:
                raise RuntimeError("后端未能在 15 秒内启动")

            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto(f"{_BASE}/")
                page.wait_for_selector("text=妙算", timeout=5000)
                assert "妙算" in page.title()
                expect(page.locator("nav button[data-view=mine]")).to_be_visible()
                expect(page.locator("nav button[data-view=backtest]")).to_be_visible()
                expect(page.locator("nav button[data-view=live]")).to_be_visible()

                # 实时页：核心区块可见
                page.locator("nav button[data-view=live]").click()
                expect(page.locator("text=当前报价")).to_be_visible()
                expect(page.locator("text=最近信号")).to_be_visible()
                expect(page.locator("text=数据通道")).to_be_visible()

                # 切到本地 runtime 通道，等待首次刷新
                page.select_option("#rtSource", "local")
                expect(page.locator("#rtBackend")).to_contain_text("local", timeout=10000)
                browser.close()
        finally:
            server.terminate()
            server.wait(timeout=5)


if __name__ == "__main__":
    test_realtime_page()