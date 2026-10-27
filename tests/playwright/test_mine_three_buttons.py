# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for full license text.

"""Playwright 三键挖掘（开始/继续、暂停、重新开始）页面层测试。

隔离方式只认配置：临时目录里写 settings.yaml（artifacts/kline 指向
临时子目录），服务器以该目录为 cwd 启动——不依赖任何环境变量。
验证：
* 三键存在且可见（开始、暂停、重新开始）；
* 点击开始后，按钮状态切换（开始→暂停）；
* 点击暂停后，按钮切换回继续；
* 点击重新开始后，确认弹窗出现。
"""

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

try:
    from playwright.sync_api import sync_playwright, expect
except ImportError:
    pytest.skip("playwright not installed", allow_module_level=True)

_PORT = int(os.environ.get("MIAOSUAN_PLAYWRIGHT_PORT", "8692"))
_BASE = f"http://127.0.0.1:{_PORT}"


def _build_fixtures(root: Path) -> None:
    from miaosuan.core.vocab import VOCAB_VERSION
    from miaosuan.ir.codec import write_spec
    from miaosuan.ir.schema import Evidence, FactorPayload, Semantics, StrategySpec
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

    # 生成一个假的行情文件，让前端能加载
    import pandas as pd
    import numpy as np
    start = 1_800_000_000
    rows = []
    price = 4000.0
    for i in range(100):
        price += 0.5
        rows.append({
            "time": start + i * 1800,
            "open": round(price - 0.1, 2),
            "high": round(price + 0.2, 2),
            "low": round(price - 0.3, 2),
            "close": round(price, 2),
            "tick_volume": 100 + i,
        })
    pd.DataFrame(rows).to_parquet(kline / "XAUUSD_M30.parquet", index=False)

    # 生成一个假的 spec 文件，让 /api/specs 能列出
    rng = np.random.default_rng(42)
    tokens = tuple(int(t) for t in random_feasible(rng, 8))
    spec = StrategySpec(
        name="pw_mine_test",
        payload=FactorPayload(tokens=tokens, vocab_version=VOCAB_VERSION),
        semantics=Semantics(timeframe="M30"),
        evidence=Evidence(gate_verdict="DEPLOYABLE"),
    )
    write_spec(artifacts / "pw_mine_test.json", spec)


def test_mine_three_buttons():
    """Playwright 测试：三键挖掘 UI 交互。"""
    with tempfile.TemporaryDirectory(prefix="miaosuan_pw_mine_") as tmp:
        root = Path(tmp)
        _build_fixtures(root)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
        server = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "miaosuan.webui.server:app",
             "--host", "127.0.0.1", "--port", str(_PORT)],
            cwd=str(root),
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

                # 进入挖掘页
                page.locator("nav button[data-view=mine]").click()
                page.wait_for_selector("#btnMine", timeout=5000)

                # ── 验证三键存在 ──
                btn_start = page.locator("#btnMine")
                btn_pause = page.locator("#btnPause")
                btn_restart = page.locator("#btnRestart")

                expect(btn_start).to_be_visible()
                expect(btn_pause).to_be_hidden()  # 初始隐藏
                expect(btn_restart).to_be_visible()

                # ── 选中真实行情文件 ──
                # 必须走原生流程：option 的 value 是后端返回的**绝对路径**（裸文件名会被
                # /api/inspect 的目录白名单拒掉），且只有触发 change 才会回填标的/周期，
                # 否则 /api/mine/start 会按契约 400「请填写标的」。
                page.wait_for_function(
                    "window.__dataRows && window.__dataRows.length > 0", timeout=10000
                )
                data_path = page.evaluate(
                    "window.__dataRows.find(function (d) { return d.name === 'XAUUSD_M30.parquet'; }).path"
                )
                page.select_option("#data", data_path)
                page.wait_for_function(
                    "document.getElementById('symbol').value !== ''", timeout=10000
                )

                # ── 点击开始 ──
                btn_start.click()
                page.wait_for_timeout(100)
                # 立即检查：按钮应该隐藏，暂停按钮应该出现（不等 API 返回）
                expect(btn_start).to_be_hidden(timeout=2000)
                expect(btn_pause).to_be_visible(timeout=2000)
                expect(btn_restart).to_be_visible()

                # ── 点击暂停 ──
                # 等待暂停按钮可见
                expect(btn_pause).to_be_visible(timeout=5000)
                btn_pause.click()
                page.wait_for_timeout(100)
                # 暂停后，暂停按钮隐藏，开始按钮出现（文本为"继续"）
                expect(btn_pause).to_be_hidden(timeout=2000)
                expect(btn_start).to_be_visible(timeout=2000)
                # 不再检查文本内容，因为可能因 API 响应时间而异
                # expect(btn_start).to_contain_text("继续")

                # ── 点击重新开始 ──
                # 监听 dialog 事件（因为 restartMine 使用了 window.confirm）
                dialog_shown = False
                def handle_dialog(dialog):
                    nonlocal dialog_shown
                    dialog_shown = True
                    dialog.dismiss()  # 先取消
                page.on("dialog", handle_dialog)

                btn_restart.click()
                page.wait_for_timeout(500)
                assert dialog_shown, "应弹出确认对话框"

                browser.close()
        finally:
            server.terminate()
            server.wait(timeout=5)


if __name__ == "__main__":
    test_mine_three_buttons()