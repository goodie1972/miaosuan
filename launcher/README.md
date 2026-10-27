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

### exe 必须留在仓库里（或显式指定根目录）

onefile 打包后，代码里的 `__file__` 指向临时解包目录（`...\Temp\_MEIxxxx`），退出 exe 该目录即被删除。
仓库根按以下顺序解析（见 `config._resolve_repo_root`）：

1. 环境变量 `MIAOSUAN_ROOT`（显式指定，优先级最高）；
2. 从 exe 所在目录向上找到工作区（判据：同层有 `settings.yaml` 与 `data/`）；
3. 从当前工作目录向上找；
4. 都找不到 → 退回当前工作目录（产物会落在那里，别再拷来拷去）。

所以默认放在 `launcher\dist\` 下双击即可；若要把 exe 单独拷走，先 `set MIAOSUAN_ROOT=D:\path\to\miaosuan`。

## License

AGPL-3.0（与主项目一致）
