#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MiaoSuan Launcher - Exact copy of working XAUUSD launcher pattern
"""
import os
import sys
import subprocess
import threading
import time
from pathlib import Path


def get_project_root():
    """获取项目根目录（无论是否被打包）"""
    if getattr(sys, 'frozen', False):
        # 如果是打包的exe
        return Path(sys.executable).parent
    else:
        # 如果是普通的Python脚本
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
    """启动后端服务 - Exact copy of XAUUSD working pattern"""
    project_root = get_project_root()
    
    # 切换到项目根目录
    os.chdir(project_root)
    
    # 确保tmp目录存在
    tmp_dir = project_root / "tmp"
    tmp_dir.mkdir(exist_ok=True)
    
    # 后端脚本路径 - 使用miaosuan.cli模块
    backend_script = "miaosuan.cli"
    
    # 日志文件路径
    log_file = tmp_dir / "launcher.log"
    
    # 将项目根目录添加到Python路径（确保能找到miaosuan模块）
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    try:
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"{'='*60}\n")
            f.write(f"MiaoSuan Trading System Launcher\n")
            f.write(f"Started at: {time.ctime()}\n")
            f.write(f"Project root: {project_root}\n")
            f.write(f"Backend script: {backend_script}\n")
            f.write(f"{'='*60}\n")
            f.flush()
        
        # 启动后端服务 - EXACT COPY OF XAUUSD PATTERN
        print("正在启动后端服务...")
        process = subprocess.Popen(
            [sys.executable, '-m', backend_script, 'ui', '--host', '127.0.0.1', '--port', '8686'],
            cwd=str(project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # 实时将输出写入日志文件 - EXACT COPY
        with open(log_file, "a", encoding="utf-8") as f:
            for line in iter(process.stdout.readline, ''):
                f.write(line)
                f.flush()
                
                # 同时显示关键信息到控制台 - EXACT COPY
                if "妙算仪表盘已启动" in line or "Uvicorn running on" in line:
                    print("✓ 后端服务已启动！")
                    # 提取URL信息
                    if "http://127.0.0.1:" in line:
                        url_part = line.split("http://127.0.0.1:")[1].split()[0]
                        print(f"  访问地址: http://127.0.0.1:{url_part}")
                elif "Application startup complete" in line:
                    print("✓ 应用程序启动完成")
        
        # 等待进程结束（实际上它应该一直运行，直到被外部终止） - EXACT COPY
        return_code = process.wait()
        
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\nBackend process ended with return code: {return_code}\n")
            f.write(f"{'='*60}\n")
            
    except Exception as e:
        error_msg = f"启动后端服务时发生错误: {e}"
        print(f"✗ {error_msg}")
        try:
            log_file = tmp_dir / "launcher_error.log"
            with open(log_file, "w", encoding="utf-8") as f:
                f.write(error_msg)
                import traceback
                f.write("\n")
                f.write(traceback.format_exc())
        except:
            pass
        raise


def main():
    """主入口函数"""
    try:
        show_banner()
        
        project_root = get_project_root()
        print(f"项目路径: {project_root}\n")
        
        # 简单检查端口是否已被占用（可选）
        print("步骤1/2: 检查现有后端...")
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('127.0.0.1', 8686))
        sock.close()
        if result == 0:
            print("  后端已经在端口 8686 运行")
            print("  正在打开浏览器...")
            import webbrowser
            webbrowser.open("http://127.0.0.1:8686")
            print("\n后端服务已在运行。")
            print("您可以继续使用此电脑进行其他操作。")
            print("要停止服务，请关闭此窗口或使用任务管理器结束相关进程。")
            if not getattr(sys, 'frozen', False):
                print("\n按 Enter 键退出启动器（后端服务将继续运行）...")
                input()
            else:
                time.sleep(3)
            return
        
        # 清理端口
        print("步骤2/2: 清理端口 8686...")
        try:
            result = subprocess.run(['netstat', '-ano'], capture_output=True, text=True, encoding='gbk', errors='ignore')
            for line in result.stdout.splitlines():
                if ':8686' in line and 'LISTENING' in line:
                    parts = line.split()
                    if len(parts) >= 5:
                        pid = parts[-1]
                        if pid.isdigit():
                            subprocess.run(['taskkill', '/F', '/PID', pid], capture_output=True)
                            print(f"  已终止 PID={pid}")
        except Exception:
            pass
        time.sleep(1)
        
        # 查找Python
        print("  查找Python解释器...")
        candidates = [
            project_root / ".venv" / "Scripts" / "python.exe",
            Path(r"C:\Python314\python.exe"),
        ]
        python_exe = None
        for c in candidates:
            if c.exists():
                python_exe = str(c)
                break
        if not python_exe:
            try:
                result = subprocess.run(['where', 'python'], capture_output=True, text=True)
                if result.returncode == 0:
                    for line in result.stdout.strip().splitlines():
                        if line.strip():
                            python_exe = line.strip()
                            break
            except Exception:
                pass
        if not python_exe:
            print("ERROR: 未找到Python解释器!")
            if not getattr(sys, 'frozen', False):
                input("\n按 Enter 键退出...")
            return 1
        print(f"  找到: {python_exe}")
        
        # 验证环境
        print("  验证环境...")
        try:
            result = subprocess.run([python_exe, '-c', 'import miaosuan; print("OK")'], capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                print(f"ERROR: 无法导入miaosuan! 调试: {python_exe} -m miaosuan.cli ui --host 127.0.0.1 --port 8686")
                if not getattr(sys, 'frozen', False):
                    input("\n按 Enter 键退出...")
                return 1
        except Exception:
            print(f"ERROR: 无法导入miaosuan! 调试: {python_exe} -m miaosuan.cli ui --host 127.0.0.1 --port 8686")
            if not getattr(sys, 'frozen', False):
                input("\n按 Enter 键退出...")
            return 1
        print("  环境验证通过")
        
        # 启动后端服务
        start_backend_service()
        
        print("\n后端服务已在后台启动。")
        print("您可以继续使用此电脑进行其他操作。")
        print("要停止服务，请关闭此窗口或使用任务管理器结束相关进程。")
        
        # 如果是打包版本且没有控制台，我们可能不需要这部分
        # 但为了调试目的，我们保持它
        if not getattr(sys, 'frozen', False):
            print("\n按 Enter 键退出启动器（后端服务将继续运行）...")
            input()
        else:
            # 对于打包的窗口版本，我们稍作停留后退出
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