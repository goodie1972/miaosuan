# 项目长期记忆

- WebUI 服务运行环境：项目本地 `.venv`（`D:/backup/BaoBao/PythonProgram/miaosuan/.venv/Scripts/python.exe`），经 `miaosuan.cli ui --host 127.0.0.1 --port 8686` 启动；新增依赖（如 tvdatafeed）必须装进这个 `.venv`，不是 managed `envs/default`。
- 数据源按「类型」划分，取值：妙算 `Shenji` / `TradingView` / `OKX` / `Binance` / `Dukascopy` / `AkShare` / `其他`；每个市场画像（`FrozenMarketProfile.data_sources`）严格匹配其可用数据源，`/api/acquisition/sources?profile=` 据此过滤。**妙算库仅存 XAUUSD，故只归属 FOREX_XAUUSD**。
- 市场画像共 **5** 个：FOREX_XAUUSD、CN_EQUITY_RESEARCH、US_EQUITY_RESEARCH、CRYPTO_BTC、CN_COMMODITY_FUTURES（商品期货，化工/农化方向）。
- fetcher 的 `is_available()` **禁止真 import 重包**（akshare / tvDatafeed 首次导入数秒会拖慢接口），改用 `importlib.util.find_spec`；网络探测须**缓存结果**并用 `getaddrinfo(AF_INET)` 单地址（否则 IPv6/IPv4 双栈各等满超时，耗时翻倍）。
- 新增市场画像会触发 `len(PROFILES)` 硬断言（tests/test_market_profiles.py 与 tests/test_core_zero_touch.py，后者 `extended` 也要 +1），必须同步修改。
- **跑 pytest 的固定姿势（必读，已踩三次）**：必须用**全新不存在**的 basetemp 目录 + 禁用 cacheprovider，例如
  `.venv\Scripts\python.exe -m pytest tests -q --basetemp .pytest_r7 -p no:cacheprovider`。
  两个参数都不可省：缺 `-p no:cacheprovider` 会因写 `.pytest_cache` 被沙箱拦下；basetemp 若指向**已存在的目录**，
  pytest 启动时对它做 `rm_rf` 同样被拦，表现为几十个 `ERROR at setup` + EXIT=1 的**假失败**（无真实用例失败）。
  另：删除文件累计超过 50 个会触发 SAFE_DELETE_BULK_REJECTED，故 `.pytest_*/` 已加进 .gitignore 而非硬删。
  基线：939 passed / 3 skipped / 0 failed。
- 本项目的系统性弱点是**静默失败**：已三度踩坑——Dukascopy `except DataError: break` 把网络不通报成"无数据"；`acquisition.py` 四处 `except Exception: pass/continue` 丢光失败原因；OKX `fetch_full` 丢弃入参 symbol 恒用默认品种（取到错数据且不报错）。**新增数据源/错误分支时，务必让失败可见**：错误向上抛、原因进消息、依赖缺失要提示怎么装。
- 依赖纪律：`fastapi`/`uvicorn` 属于 `web` extra，`tvdatafeed`/`akshare`/`tqsdk`/`websocket-client` 属于 `datasource` extra，均在 pyproject 显式声明；惰性 import 缺失时 `describe()` 必须给可行动提示，禁止静默返回 False。env 读取只允许在 `config.py`/`cli.py`，新增硬编码路径要配套 accessor（如 `MIAOSUAN_KLINE_DIR`）。
- **Spec 管理方案**（2026-09-25 设计，文档 `docs/spec-management.md`）：核心设计为新增 `SpecRegistry`（`artifacts/spec_registry.json`）作为独立索引层，不侵入 spec 四段结构。Provenance 计划新增 `parent_spec_id`/`lineage`/`derivation` 三字段（向后兼容）。当前缺口：无 spec 索引/注册表、无谱系追踪、无 spec↔.py 反向映射、tune 不产出派生 spec。实施分四阶段：registry 基础 → 谱系追踪 → 回测/tune 记录 → 导出策略与自动化。

## 2026-09-25

- Cleaned up unused files: removed artifacts/_legacy_tmp/ (23 files), artifacts/qa/ (416 files), root *.log files, tmp launcher logs, and pytest temp directories. Artifacts/ now only contains .gitkeep.
- Fixed and enhanced one-click launcher start_miaosuan.bat to v1.3:
    * Uses cd /d "%~dp0" to eliminate working directory dependence
    * Generates a temporary backend runner script with absolute-path awareness
    * Launches backend in a minimized independent window via start /MIN
    * Preserves original functionality: health check, port cleanup, Python detection, readiness polling, browser auto-open
- Confirmed direct backend launch works: python -m miaosuan.cli ui --host 127.0.0.1 --port 8686 starts the WebUI and serves /api/meta correctly.
- The Spec Management UI module is fully implemented per earlier design (nav button, detail view, lineage tree, export/backtest/tune tables, CSS adjustments).
- **2026-09-25 更新**：最终启动器解决方案
    * launcher.py 简化为直接复制神机启动器的工作模式（主线程读取输出，保持管道打开）
    * bat 文件用 `.venv\Scripts\python.exe launcher.py`（**不能用裸 `python`**，否则系统 PATH 解析到 C:\Python314 找不到 miaosuan 模块）
    * launcher.py 内部用 `sys.executable` 启动后端子进程，所以 bat 调用的 Python 必须是 venv 的
    * 移除了所有HTTP轮询和复杂逻辑，专注于让主线程阻塞在读取子进程输出上
    * 修复后日志确认正常启动："妙算仪表盘已启动：http://127.0.0.1:8686"

## 2026-09-26

- **统一配置系统落地**（完成用户需求：可配置设置模块、动态配置持久化、解决硬编码端口、热重载）：
  - 新建 `src/miaosuan/settings.py`：Pydantic 模型验证、YAML + 环境变量（`MIAOSUAN_*`）分层、watchdog 文件监听热重载、运行时覆盖接口、向后兼容旧版环境变量名（`MIAOSUAN_MT4_BRIDGE_HOST` 等）。
  - 新建 `settings.yaml`：包含 webui、shenji、mt4、paths、data、logging、features 七大配置节，全部保留原硬编码默认值（webui=8686、shenji=1783、mt4=23232、kline=D:\K线数据 等）。
  - 重构 `src/miaosuan/config.py` 为适配层：保持 `shenji_backend_config`/`data_acquisition_config`/`kline_data_dir`/`is_bundled` 等全部公开签名不变，内部委托 `settings.get_config()`；环境变量实时读取 `os.environ`，兼容 pytest `monkeypatch` 测试；`AppConfig` 等核心配置类保留，核心层依赖注入零改动。
  - 硬编码迁移：`cli.py`（ui 端口）、`mt4_bridge.py`（MT4 端口）、`acquisition.py`（K 线目录）、`launcher.py`（端口/主机）均改为运行时从设置系统读取。
  - 验证：939 passed / 3 skipped / 0 failed；启动器正常工作；MT4 fetcher 默认端口 23232 可通过 `settings.yaml` 修改即时生效（重启进程或热重载）。
