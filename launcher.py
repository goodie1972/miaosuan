#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MiaoSuan Launcher - 使用统一配置系统

启动方式：.venv\\Scripts\\python.exe launcher.py
（bat 文件已确保用 venv Python 调用本脚本，故 sys.executable 即 venv 解释器）
"""
import os
import sys
import subprocess
import time
from pathlib import Path

# 导入统一配置系统
sys.path.insert(0, str(Path(__file__).parent / "src"))
from miaosuan.settings import get_config


def get_project_root():
    """获取项目根目录（无论是否被打包）"""
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).parent
    else:
        return Path(__file__).parent.absolute()


def show_banner():
    """显示启动横幅"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║          MiaoSuan 量化交易系统                              ║
    ║                                                              ║
    ║          专业量化交易平台启动器                            ║
    ║                                                              ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def start_backend_service():
    """启动后端服务 - 使用配置系统获取端口"""
    project_root = get_project_root()

    # 切换到项目根目录（settings.yaml 在这里）
    os.chdir(project_root)

    # 确保tmp目录存在
    tmp_dir = project_root / "tmp"
    tmp_dir.mkdir(exist_ok=True)

    # 后端脚本路径
    backend_script = "miaosuan.cli"

    # 获取配置中的端口和主机
    config = get_config()
    port = config.webui.port
    host = config.webui.host

    # 日志文件路径
    log_file = tmp_dir / "launcher.log"

    # 将项目根目录添加到Python路径
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"{'='*60}\n")
            f.write(f"MiaoSuan Trading System Launcher\n")
            f.write(f"Started at: {time.ctime()}\n")
            f.write(f"Project root: {project_root}\n")
            f.write(f"Backend script: {backend_script}\n")
            f.write(f"WebUI: http://{host}:{port}\n")
            f.write(f"Python: {sys.executable}\n")
            f.write(f"{'='*60}\n")
            f.flush()

        # 启动后端服务 - sys.executable 即 venv Python（bat 文件保证）
        print(f"正在启动后端服务 (端口 {port})...")
        process = subprocess.Popen(
            [sys.executable, '-m', backend_script, 'ui', '--host', str(host), '--port', str(port)],
            cwd=str(project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # 实时将输出写入日志文件（阻塞式读取，保持管道打开）
        with open(log_file, "a", encoding="utf-8") as f:
            for line in iter(process.stdout.readline, ''):
                f.write(line)
                f.flush()

                # 显示关键信息到控制台
                if "妙算仪表盘已启动" in line or "Uvicorn running on" in line:
                    print("[OK] 后端服务已启动")
                    # 尝试自动打开浏览器
                    try:
                        import webbrowser
                        webbrowser.open(f"http://{host}:{port}")
                    except Exception:
                        pass
                elif "Application startup complete" in line:
                    print("[OK] 应用程序启动完成")

        # 等待进程结束
        return_code = process.wait()

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\nBackend process ended with return code: {return_code}\n")
            f.write(f"{'='*60}\n")

    except Exception as e:
        error_msg = f"启动后端服务时发生错误: {e}"
        print(f"[ERROR] {error_msg}")
        try:
            log_file = tmp_dir / "launcher_error.log"
            with open(log_file, "w", encoding="utf-8") as f:
                f.write(error_msg)
                import traceback
                f.write("\n")
                f.write(traceback.format_exc())
        except Exception:
            pass
        raise


def find_port_pids(port: int) -> list[int]:
    """找出占用指定端口的所有进程 PID（netstat 解析，仅 LISTENING 状态）。"""
    pids: list[int] = []
    try:
        result = subprocess.run(['netstat', '-ano'], capture_output=True, text=True,
                                encoding='gbk', errors='ignore')
        for line in result.stdout.splitlines():
            if f':{port}' in line and 'LISTENING' in line:
                parts = line.split()
                if len(parts) >= 5 and parts[-1].isdigit():
                    pid = int(parts[-1])
                    if pid and pid not in pids:
                        pids.append(pid)
    except Exception:
        pass
    return pids


def kill_stale_backend(port: int) -> bool:
    """终止占用端口的旧后端进程并等待端口释放。

    返回 True 表示端口已可用（原本空闲或旧进程已终止）。
    旧进程不终止的话，launcher 只会看到「已在运行」，用户改完代码重启
    也加载不到新代码——这正是「重启了还是不对」的根因。
    """
    pids = find_port_pids(port)
    if not pids:
        return True
    my_pid = os.getpid()
    for pid in pids:
        if pid == my_pid:
            continue  # 不会发生，但防御一下：绝不杀自己
        print(f"  发现旧后端进程 PID={pid}，正在终止…")
        try:
            subprocess.run(['taskkill', '/F', '/T', '/PID', str(pid)],
                           capture_output=True)
        except Exception as exc:
            print(f"  终止失败（{exc}），请手动在任务管理器结束该进程。")
            return False
    # 等待端口真正释放（TIME_WAIT/句柄回收需要一两秒）
    for _ in range(10):
        if not find_port_pids(port):
            return True
        time.sleep(0.5)
    print(f"  端口 {port} 仍被占用，请手动处理后重试。")
    return False


def main():
    """主入口函数"""
    try:
        show_banner()

        project_root = get_project_root()
        print(f"项目路径: {project_root}\n")

        # 获取配置
        config = get_config()
        port = config.webui.port
        host = config.webui.host

        # 检查端口是否已被占用
        print(f"步骤 1/2: 检查端口 {port}...")
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex((str(host), port))
        sock.close()
        if result == 0:
            # 端口被占 = 旧后端还活着。直接终止它再启动新后端，
            # 否则「重启」永远加载不到新代码（用户改完配置/代码重启无效的根因）。
            print(f"  发现旧后端仍在运行（端口 {port}）")
            if not kill_stale_backend(port):
                if not getattr(sys, 'frozen', False):
                    input("\n按 Enter 键退出...")
                else:
                    time.sleep(3)
                return 1
        time.sleep(1)

        # 验证当前 Python 能导入 miaosuan
        print("  验证环境...")
        try:
            result = subprocess.run(
                [sys.executable, '-c', 'import miaosuan; print("OK")'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                print(f"ERROR: 无法导入 miaosuan!")
                print(f"  Python: {sys.executable}")
                print(f"  调试: {sys.executable} -m miaosuan.cli ui --host {host} --port {port}")
                if not getattr(sys, 'frozen', False):
                    input("\n按 Enter 键退出...")
                return 1
        except Exception:
            print(f"ERROR: 环境验证失败")
            print(f"  Python: {sys.executable}")
            if not getattr(sys, 'frozen', False):
                input("\n按 Enter 键退出...")
            return 1
        print("  环境验证通过")

        # 启动后端服务
        start_backend_service()

        print("\n后端服务已启动。")
        print("要停止服务，请关闭此窗口或使用任务管理器结束相关进程。")

        if not getattr(sys, 'frozen', False):
            print("\n按 Enter 键退出启动器（后端服务将继续运行）...")
            input()
        else:
            time.sleep(3)

    except KeyboardInterrupt:
        print("\n\n用户中断启动过程。")
    except Exception as e:
        print(f"\n启动失败: {e}")
        print("请查看 tmp/launcher_error.log 获取详细信息。")
        if not getattr(sys, 'frozen', False):
            input("按 Enter 键退出...")


if __name__ == "__main__":
    main()
