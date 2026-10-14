#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for full license text.

"""e2e 冒烟脚本（路线图 R5.3）：本地 runtime 通道 + 通知装配。

流程：临时目录造数据（缓存 K 线 + spec + watchlist + **settings.yaml**）→
以该目录为 cwd 起真实服务（**只认配置**：artifacts/kline 全部来自临时
settings.yaml，不依赖任何环境变量）→ ``source=local`` 四个端点返回
**真实本地状态** → 通知器按配置装配 → 断言。

不依赖真实 MT5 终端与神机后台；全部用临时数据与本地服务。
用法：``MIAOSUAN_E2E_PORT=8791 .venv/Scripts/python.exe scripts/test_e2e_realtime.py``
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

#: 端口可用环境变量覆盖（``MIAOSUAN_E2E_PORT``），避免与已在跑的实例冲突。
_E2E_PORT = int(os.environ.get("MIAOSUAN_E2E_PORT", "8686"))
API_BASE = f"http://127.0.0.1:{_E2E_PORT}"
SERVER_READY_TIMEOUT = 30.0


def _build_fixtures(root: Path) -> dict[str, Path]:
    """造临时数据：settings.yaml + 缓存 K 线 + DEPLOYABLE spec + watchlist 条目。

    全部写入 ``root`` 下的子目录；``root/settings.yaml`` 把它们登记为
    ``paths.artifacts`` / ``paths.kline``——服务器以 ``root`` 为 cwd 启动
    时只从这份配置解析目录（只认配置，无 env 回退）。
    """
    from miaosuan.core.vocab import VOCAB_VERSION
    from miaosuan.ir.codec import write_spec
    from miaosuan.ir.schema import Evidence, FactorPayload, Semantics, StrategySpec
    from miaosuan.runtime.watchlist import WatchEntry, WatchlistStore
    from miaosuan.search.rpn import random_feasible

    artifacts = root / "artifacts"
    kline = root / "kline"
    artifacts.mkdir(parents=True, exist_ok=True)
    kline.mkdir(parents=True, exist_ok=True)

    # 配置：artifacts/kline 全部指向临时目录（只认配置的隔离手段）。
    (root / "settings.yaml").write_text(
        "paths:\n"
        f"  artifacts: {artifacts.as_posix()}\n"
        f"  kline: {kline.as_posix()}\n"
        "notifiers:\n"
        "  feishu:\n"
        "    webhook_url: ''\n",
        encoding="utf-8",
    )

    # 缓存 K 线（520 根，超过默认 warmup=499，信号引擎才会真正计算特征）。
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
        name="e2e_local_smoke",
        payload=FactorPayload(tokens=tokens, vocab_version=VOCAB_VERSION),
        semantics=Semantics(timeframe="M30"),
        evidence=Evidence(gate_verdict="DEPLOYABLE"),
    )
    spec_path = artifacts / "e2e_local_smoke.json"
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
    return {"artifacts": artifacts, "kline": kline, "root": root}


def _start_server(root: Path) -> subprocess.Popen[bytes]:
    """以临时目录为 cwd 启动 uvicorn（cwd 含 settings.yaml → 只认配置）。"""
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "miaosuan.webui.server:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(_E2E_PORT),
        ],
        cwd=str(root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + SERVER_READY_TIMEOUT
    while time.time() < deadline:
        try:
            if requests.get(f"{API_BASE}/api/health", timeout=1).status_code == 200:
                return proc
        except requests.RequestException:
            time.sleep(0.5)
    proc.terminate()
    raise RuntimeError(f"服务未能在 {SERVER_READY_TIMEOUT:.0f} 秒内启动")


def _stop_server(proc: subprocess.Popen[bytes]) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def _assert_local_channel() -> None:
    # 1) 引擎状态：watchlist 条目 + 门禁 verdict（真实只读）
    st = requests.get(
        f"{API_BASE}/api/realtime/status", params={"source": "local"}, timeout=3
    ).json()
    assert st["ok"] is True and st["backend"] == "local", st
    wl = st["data"]["watchlist"]
    assert len(wl) == 1 and wl[0]["symbol"] == "XAUUSD", wl
    assert wl[0]["gate_verdict"] == "DEPLOYABLE", wl
    print("✅ 本地状态：watchlist + 门禁 verdict")

    # 2) K 线：真实收盘价（非 0.0 假蜡烛）
    cd = requests.get(
        f"{API_BASE}/api/realtime/candles",
        params={"source": "local", "symbol": "XAUUSD", "timeframe": "M30", "count": 5},
        timeout=3,
    ).json()
    assert cd["ok"] is True and len(cd["data"]) == 5, cd
    closes = [c["close"] for c in cd["data"]]
    assert min(closes) > 4000, closes
    print(f"✅ 本地 K 线：真实收盘价 {closes}")

    # 3) 报价：缓存末根收盘价 + 显式来源
    pr = requests.get(
        f"{API_BASE}/api/realtime/price",
        params={"source": "local", "symbol": "XAUUSD"},
        timeout=3,
    ).json()
    assert pr["ok"] is True and pr["data"]["source"] == "cached_last_close", pr
    assert pr["data"]["bid"] > 4000, pr
    print(f"✅ 本地报价：{pr['data']['bid']}（source=cached_last_close）")

    # 4) 信号：真实仓位/方向（引擎跑通，不是 unavailable 桩）
    sg = requests.get(
        f"{API_BASE}/api/realtime/signals", params={"source": "local"}, timeout=3
    ).json()
    assert sg["ok"] is True and len(sg["data"]) == 1, sg
    row = sg["data"][0]
    assert row["status"] == "signal", row
    assert row["strategy"] == "e2e_local_smoke", row
    print(f"✅ 本地信号：{row['strategy']} dir={row['signal']} pos={row['position']}")


def _assert_notifier_wiring() -> None:
    """通知装配只认配置：空 webhook → 仅日志；非空 → 多通道（含飞书）。"""
    from miaosuan.runtime.notifiers import MultiNotifier, build_notifier
    from miaosuan.settings import Settings

    cfg = Settings()
    import miaosuan.settings as settings_mod

    orig = settings_mod.get_config
    try:
        cfg.notifiers.feishu.webhook_url = ""
        settings_mod.get_config = lambda: cfg
        assert not isinstance(build_notifier(), MultiNotifier)
        cfg.notifiers.feishu.webhook_url = "https://open.feishu.cn/hook/x"
        n = build_notifier()
        assert isinstance(n, MultiNotifier), type(n)
        assert "FeishuNotifier" in [type(ch).__name__ for ch in n.channels]
    finally:
        settings_mod.get_config = orig
    print("✅ 通知装配：空=仅日志，非空=日志+飞书")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="miaosuan_e2e_") as tmp:
        root = Path(tmp)
        fx = _build_fixtures(root)
        print("启动服务 …")
        server = _start_server(fx["root"])
        try:
            print("校验本地 runtime 通道 …")
            _assert_local_channel()
            print("校验通知装配 …")
            _assert_notifier_wiring()
            print("✅ e2e 冒烟通过：本地通道真实数据 + 通知装配正确")
        finally:
            print("停止服务 …")
            _stop_server(server)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - 冒烟脚本：失败即非零退出
        print(f"❌ e2e 冒烟失败：{exc}")
        sys.exit(1)