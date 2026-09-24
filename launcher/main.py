#!/usr/bin/env python3
"""妙算一键启动器——双击即用，无控制台窗口。

两种模式：
1. 无参数（双击启动）→ 启动 uvicorn 内嵌服务 + 打开浏览器 + MessageBox 守护
2. 带 CLI 子命令（mine/export/verify/...）→ 路由到 miaosuan.cli

设计要点：
- 不依赖 tkinter（.venv Python 3.13 无 tkinter）
- 使用 ctypes Win32 MessageBoxW 做守护窗口（关闭即退出）
- uvicorn 运行在 daemon 线程中，os._exit(0) 强制终止
"""

from __future__ import annotations

import contextlib
import ctypes
import os
import socket
import sys
import threading
import time
import webbrowser

# ── 常量 ──────────────────────────────────────────────────────────────

HOST = "127.0.0.1"
PORT = 8686
READY_TIMEOUT = 30  # 等待后端启动的最长时间（秒）
CHECK_INTERVAL = 0.5  # 轮询间隔

# 已知 CLI 子命令——exe 被当作 CLI 调用时据此路由
_CLI_COMMANDS = {"mine", "export", "verify", "report", "backtest", "tune", "ui"}


# ── 工具函数 ──────────────────────────────────────────────────────────


def _say(msg: str) -> None:
    """windowed exe 下 sys.stdout 可能为 None，统一安全输出。"""
    with contextlib.suppress(Exception):
        print(msg, flush=True)


def _is_port_open(port: int = PORT, host: str = HOST) -> bool:
    """检查 TCP 端口是否已监听。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.settimeout(1.0)
            s.connect((host, port))
            return True
        except Exception:
            return False


def _is_bundled() -> bool:
    """True when running inside a PyInstaller bundle."""
    return getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")


# ── CLI 路由 ──────────────────────────────────────────────────────────


def _run_cli(args: list[str]) -> int:
    """当 exe 带子命令参数被调用时，路由到 miaosuan.cli。

    在 PyInstaller bundle 中，server.py 的 _cli_process() 会以
    [sys.executable, *args] 形式 spawn 本 exe，本函数接管 CLI 执行。
    """
    from miaosuan.cli import app

    sys.argv = [sys.argv[0]] + list(args)
    try:
        app()
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 0
    return 0


# ── 内嵌 uvicorn 服务 ─────────────────────────────────────────────────


def _start_server() -> None:
    """在当前进程内启动 uvicorn（非子进程），用于 bundle 模式。"""
    import uvicorn

    from miaosuan.webui.server import create_app

    config = uvicorn.Config(
        create_app(),
        host=HOST,
        port=PORT,
        log_level="warning",
    )
    server = uvicorn.Server(config)
    # 阻止 uvicorn 安装信号处理器——我们在主线程控制退出
    server.install_signal_handlers = lambda: None
    server.run()


# ── 启动器主逻辑 ──────────────────────────────────────────────────────


def _launcher_mode() -> int:
    """启动 uvicorn 服务（daemon 线程），打开浏览器，MessageBox 守护。"""
    # 标记 bundle 模式——server.py 的 _cli_process() 据此切换调用方式
    os.environ["MIAOSUAN_BUNDLED"] = "1"

    # 在 daemon 线程中启动 uvicorn
    server_thread = threading.Thread(target=_start_server, daemon=True)
    server_thread.start()
    _say("后端服务线程已启动，等待端口就绪...")

    # 轮询端口
    start = time.time()
    ready = False
    while time.time() - start < READY_TIMEOUT:
        if _is_port_open():
            ready = True
            break
        time.sleep(CHECK_INTERVAL)

    if not ready:
        _say("启动超时")
        ctypes.windll.user32.MessageBoxW(
            0,
            f"启动超时：后端服务在 {READY_TIMEOUT} 秒内未就绪。\n请检查端口 {PORT} 是否被占用。",
            "妙算仪表盘 - 启动失败",
            0x00000010,  # MB_ICONERROR
        )
        os._exit(1)

    # 打开浏览器
    url = f"http://{HOST}:{PORT}"
    _say(f"服务就绪，打开浏览器: {url}")
    webbrowser.open_new_tab(url)

    # MessageBox 守护——用户点击「确定」后退出
    ctypes.windll.user32.MessageBoxW(
        0,
        f"妙算仪表盘正在运行中。\n\n浏览器已打开: {url}\n\n点击「确定」关闭服务并退出。",
        "妙算仪表盘",
        0x00000040,  # MB_ICONINFORMATION
    )

    # 用户已确认退出——强制终止（daemon 线程会被杀死）
    _say("用户确认退出，终止服务...")
    os._exit(0)


# ── 入口 ──────────────────────────────────────────────────────────────


def main() -> int:
    args = sys.argv[1:]

    # 如果带已知 CLI 子命令，路由到 CLI 模式
    if args and args[0] in _CLI_COMMANDS:
        return _run_cli(args)

    # 否则进入启动器模式
    return _launcher_mode()


if __name__ == "__main__":
    sys.exit(main())
