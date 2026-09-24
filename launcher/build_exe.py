#!/usr/bin/env python3
r"""妙算一键启动器打包脚本。

用法：
    .venv\Scripts\python.exe build_exe.py

产出：dist/miaosuan_launcher.exe
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LAUNCHER_DIR = Path(__file__).resolve().parent
ENTRY = LAUNCHER_DIR / "main.py"
OUTPUT_NAME = "miaosuan_launcher"

# 使用项目 .venv Python（已安装所有依赖 + PyInstaller）
PYTHON_EXE = REPO_ROOT / ".venv" / "Scripts" / "python.exe"

ARGS = [
    str(PYTHON_EXE),
    "-m",
    "PyInstaller",
    "--onefile",
    "--noconsole",
    "--name",
    OUTPUT_NAME,
    "--distpath",
    str(LAUNCHER_DIR / "dist"),
    "--workpath",
    str(LAUNCHER_DIR / "build"),
    "--specpath",
    str(LAUNCHER_DIR),
    # 收集所有 miaosuan 子模块（editable install 需要显式收集）
    "--collect-all",
    "miaosuan",
    # uvicorn/fastapi 隐式依赖
    "--hidden-import",
    "uvicorn",
    "--hidden-import",
    "uvicorn.logging",
    "--hidden-import",
    "uvicorn.loops",
    "--hidden-import",
    "uvicorn.loops.auto",
    "--hidden-import",
    "uvicorn.protocols",
    "--hidden-import",
    "uvicorn.protocols.http",
    "--hidden-import",
    "uvicorn.protocols.http.auto",
    "--hidden-import",
    "uvicorn.protocols.websockets",
    "--hidden-import",
    "uvicorn.protocols.websockets.auto",
    "--hidden-import",
    "uvicorn.lifespan",
    "--hidden-import",
    "uvicorn.lifespan.on",
    "--hidden-import",
    "fastapi",
    "--hidden-import",
    "fastapi.responses",
    "--hidden-import",
    "fastapi.staticfiles",
    # 排除重型可选依赖（打包时不需要，运行时缺失会在 describe() 中提示用户安装）
    "--exclude-module",
    "tvdatafeed",
    "--exclude-module",
    "akshare",
    "--exclude-module",
    "tqsdk",
    "--exclude-module",
    "websocket",
    "--exclude-module",
    "websocket-client",
    "--exclude-module",
    "tkinter",
    "--exclude-module",
    "matplotlib",
    "--exclude-module",
    "PIL",
    # 不使用 UPX（可能破坏 DLL）
    "--noupx",
    str(ENTRY),
]


def main() -> None:
    print(f"Building {OUTPUT_NAME}.exe")
    print(f"Python: {PYTHON_EXE}")
    print(f"Entry: {ENTRY}")
    print(f"Repo root: {REPO_ROOT}")
    print()

    result = subprocess.run(ARGS, cwd=str(REPO_ROOT))
    if result.returncode == 0:
        exe_path = LAUNCHER_DIR / "dist" / f"{OUTPUT_NAME}.exe"
        print(f"\n✅ Build successful: {exe_path}")
        print(f"   Size: {exe_path.stat().st_size / 1024 / 1024:.1f} MB")
    else:
        print(f"\n❌ Build failed with code {result.returncode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
