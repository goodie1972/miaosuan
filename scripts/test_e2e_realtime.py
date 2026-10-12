#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for full license text.

"""e2e 冒烟脚本（路线图 R5.3）：
启动服务 → watch 添加条目 → 收盘信号产出 → 本地通道返回信号 → 断言。
此脚本不依赖真实 MT5 终端；全部使用假装置。
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests

# ─────────────────────────────────────────────────────────────────────────────
# 辅助：启动/停止服务
# ─────────────────────────────────────────────────────────────────────────────

API_BASE = "http://127.0.0.1:8686"


def start_server() -> subprocess.Popen[Any]:
    env = os.environ.copy()
    env["MIAOSUAN_ARTIFACTS_DIR"] = str(Path("tests/test_adversarial/data"))  # 利用已有测试数据
    # 使用 uvicorn 直接启动服务
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "miaosuan.webui.server:app",
         "--host", "127.0.0.1", "--port", "8686"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # 等待服务就绪
    for _ in range(30):
        try:
            r = requests.get(f"{API_BASE}/api/health", timeout=1)
            if r.status_code == 200:
                break
        except Exception:
            time.sleep(0.5)
    else:
        proc.terminate()
        raise RuntimeError("服务未能在 15 秒内启动")
    return proc


def stop_server(proc: subprocess.Popen[Any]) -> None:
    proc.terminate()
    proc.wait(timeout=5)


# ─────────────────────────────────────────────────────────────────────────────
# 辅助：创建 watch 条目
# ─────────────────────────────────────────────────────────────────────────────

def add_watch_item(symbol: str = "XAUUSD", timeframe: str = "M30") -> None:
    # 先生成一个可用的 spec（放进 artifacts/）
    from miaosuan.core.vocab import VOCAB_VERSION
    from miaosuan.ir.schema import Evidence, FactorPayload, Semantics, StrategySpec
    from miaosuan.ir.codec import write_spec

    rng = __import__("numpy").random.default_rng(42)
    tokens = tuple(int(t) for t in __import__("miaosuan.search.rpn").random_feasible(rng, 8))
    spec = StrategySpec(
        name="e2e_smoke",
        payload=FactorPayload(tokens=tokens, vocab_version=VOCAB_VERSION),
        semantics=Semantics(timeframe=timeframe),
        evidence=Evidence(gate_verdict="DEPLOYABLE"),
    )
    artifacts_dir = Path(os.environ.get("MIAOSUAN_ARTIFACTS_DIR", "artifacts"))
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    spec_path = artifacts_dir / "e2e_smoke.json"
    write_spec(spec_path, spec)

    # 再把条目加到 watchlist
    from miaosuan.runtime.watchlist import WatchlistStore
    store = WatchlistStore()
    store.clear()
    store.add(
        {
            "symbol": symbol,
            "spec_path": str(spec_path),
            "profile": "FOREX_XAUUSD_MT5",  # 现有画像
            "timeframe": timeframe,
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("启动服务 …")
    server = start_server()
    try:
        print("添加 watch 条目 …")
        add_watch_item()

        print("等待首条收盘信号（最多 60 秒） …")
        deadline = time.time() + 60
        signal_val = None
        while time.time() < deadline:
            try:
                r = requests.get(f"{API_BASE}/api/realtime/signals", timeout=2)
                if r.status_code == 200:
                    data = r.json()
                    if data.get("signals"):
                        signal_val = data["signals"][0].get("signal")
                        if signal_val is not None:
                            print(f"收到信号：{signal_val}")
                            break
            except Exception:
                pass
            time.sleep(1)
        else:
            raise RuntimeError("在 60 秒内未收到任何信号")

        # 断言：信号应为数值（非 None）
        assert isinstance(signal_val, (int, float)), f"信号应为数值，实际得到 {signal_val!r}"
        print("✅ e2e 冒烟通过：服务可启动、watch 条目生效、本地实时信号可查询")
    finally:
        print("停止服务 …")
        stop_server(server)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ e2e 冒烟失败：{e}")
        sys.exit(1)