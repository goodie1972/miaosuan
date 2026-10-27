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


def _anchor_bundled_env() -> None:
    """把「仓库根」与「我是打包 exe」两件事固化进环境，供本进程与子进程共用。

    - ``MIAOSUAN_ROOT``：onefile 下 ``__file__`` 指向临时解包目录（``...\\Temp\\_MEIxxxx``），
      靠代码位置反推仓库根会指到 Temp——spec / 策略 / 结果列表因此在 exe 里全空，且退出后
      产物随临时目录一起被删。config 里已做同样的搜索，这里把结论钉进环境，子进程直接继承。
    - ``MIAOSUAN_BUNDLED``：必须在 **frozen 时**就置位，而不是只在 :func:`_launcher_mode`
      里置位。``server._cli_process()`` 据此选 spawn 形态：打包态 ``[exe, "verify", ...]``
      （由 :func:`_run_cli` 路由），源码态 ``[python, "-m", "miaosuan.cli", ...]``。
      只在启动器模式置位时，用 ``exe ui --port x``（CLI 模式起仪表盘）会 spawn 出
      ``exe -m miaosuan.cli verify ...``——exe 认不出这串参数，落回启动器模式再起一个服务
      并弹 MessageBox，「验证」按钮因此永久卡死。
    """
    if not (os.environ.get("MIAOSUAN_ROOT", "") or "").strip():
        from miaosuan.config import REPO_ROOT

        os.environ["MIAOSUAN_ROOT"] = str(REPO_ROOT)
    if getattr(sys, "frozen", False):
        os.environ["MIAOSUAN_BUNDLED"] = "1"


def main() -> int:
    _anchor_bundled_env()
    args = sys.argv[1:]

    # 带已知 CLI 子命令 → 路由到 CLI 模式
    if args and args[0] in _CLI_COMMANDS:
        return _run_cli(args)

    # 打包态拿到不认识的参数（如上游又按源码形态 spawn 出 ``-m miaosuan.cli ...``）：
    # 必须就地失败。落回启动器模式会在用户眼前弹 MessageBox、再起一个占用 8686 的服务，
    # 而调用方等到超时——把「明确的退出码 2」交给 UI 显示。
    if args and getattr(sys, "frozen", False):
        _say(f"未知命令：{args[0]!r}；可用命令：{sorted(_CLI_COMMANDS)}")
        return 2

    return _launcher_mode()


if __name__ == "__main__":
    sys.exit(main())
