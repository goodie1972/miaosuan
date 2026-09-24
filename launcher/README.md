# 妙算一键启动器

双击即可启动妙算 Web 仪表盘并自动打开浏览器（`http://127.0.0.1:8686`）。

## 方式一：Node 启动器（推荐，已验证可用）

```bash
node launcher/run.js
```

行为：
1. 探测 `127.0.0.1:8686` 是否已有后端在跑（有则直接开浏览器）
2. 无则清理占用端口的残留进程，启动 `python -m miaosuan.cli ui`
3. 轮询 `/api/meta` 等待就绪（最多 30 秒），随后打开默认浏览器
4. 后端日志写入 `tmp/launcher_backend.log`

Python 解释器优先使用项目 `.venv\Scripts\python.exe`，找不到时回退系统 Python。

## 方式二：PyInstaller 打包为 exe（可选）

```bash
.venv\Scripts\python.exe launcher\build_exe.py
# 产出 launcher\dist\miaosuan_launcher.exe
```

- 入口：`launcher/main.py`（无参数 = 启动服务 + MessageBox 守护；带 CLI 子命令 = 路由到 `miaosuan.cli`）
- 打包为单文件、无控制台窗口的 exe；服务运行期间显示 MessageBox，点「确定」即退出

## License

AGPL-3.0（与主项目一致）
