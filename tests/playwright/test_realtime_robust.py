# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for full license text.

"""Playwright test for the webui realtime page (R5.1-R5.2) - robust version."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

try:
    from playwright.sync_api import sync_playwright, expect
except ImportError:
    pytest.skip("playwright not installed", allow_module_level=True)


def test_realtime_page():
    # 创建临时目录用于 artifacts
    import tempfile
    tmpdir = Path(tempfile.mkdtemp())
    env = os.environ.copy()
    env["MIAOSUAN_ARTIFACTS_DIR"] = str(tmpdir)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
    # 启动后端（uvicorn）
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "miaosuan.webui.server:app",
         "--host", "127.0.0.1", "--port", "8686"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        # 等待服务就绪
        for _ in range(30):
            try:
                import requests
                r = requests.get("http://127.0.0.1:8686/api/health", timeout=1)
                if r.status_code == 200:
                    print("Server is up")
                    break
            except Exception as e:
                print(f"Health check failed: {e}")
            time.sleep(0.5)
        else:
            server.terminate()
            server.wait(timeout=5)
            raise RuntimeError("服务未能在 15 秒内启动")

        # 创建一个简单的 spec 并添加到 watchlist
        from miaosuan.core.vocab import VOCAB_VERSION
        from miaosuan.ir.schema import Evidence, FactorPayload, Semantics, StrategySpec
        from miaosuan.ir.codec import write_spec

        rng = __import__("numpy").random.default_rng(42)
        from miaosuan.search.rpn import random_feasible
        tokens = tuple(int(t) for t in random_feasible(rng, 8))
        spec = StrategySpec(
            name="test_realtime",
            payload=FactorPayload(tokens=tokens, vocab_version=VOCAB_VERSION),
            semantics=Semantics(timeframe="M30"),
            evidence=Evidence(gate_verdict="DEPLOYABLE"),
        )
        spec_path = tmpdir / "test_realtime.json"
        write_spec(spec_path, spec)

        from miaosuan.runtime.watchlist import WatchlistStore, WatchEntry
        store_path = tmpdir / "watchlist.json"
        store = WatchlistStore(store_path)
        # Clear the watchlist by saving an empty list
        store.save([])
        store.add(
            WatchEntry(
                symbol="XAUUSD",
                spec_path=str(spec_path),
                profile="FOREX_XAUUSD_MT5",
                timeframe="M30",
            )
        )

# 启动 Playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto("http://127.0.0.1:8686/")
            # 等待页面加载
            page.wait_for_selector("text=妙算", timeout=10000)
            # 断言页面标题
            assert "妙算" in page.title()
            # 可以进一步断言页面中存在某些元素，例如导航栏
            expect(page.locator("nav button[data-view=mine]")).to_be_visible()
            expect(page.locator("nav button[data-view=backtest]")).to_be_visible()
            expect(page.locator("nav button[data-view=live]")).to_be_visible()
            # 点击实时标签页并检查实时页面内容
            page.locator("nav button[data-view=live]").click()
            expect(page.locator("text=当前报价")).to_be_visible()
            expect(page.locator("text=最近信号")).to_be_visible()
            browser.close()
    finally:
        server.terminate()
        server.wait(timeout=5)
        # 打印服务器日志以供调试
        stdout, stderr = server.communicate()
        if stdout:
            print("Server stdout:", stdout.decode())
        if stderr:
            print("Server stderr:", stderr.decode())
        # 清理临时目录
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    test_realtime_page()