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
