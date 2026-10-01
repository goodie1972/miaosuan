# Dukascopy 历史数据全周期拉取工具

独立编译的命令行工具，从 Dukascopy 公开 API 拉取 XAUUSD（或其他品种）的全历史 K 线数据，输出为标准 Parquet 文件。

---

## 文件清单

| 文件 | 说明 |
|------|------|
| `dist/duk_fetch.exe` | 编译好的可执行文件（约 102 MB，Windows x64） |
| `duk_fetch_app.py` | 源码（自包含，可直接用 Python 运行） |
| `duk_fetch_app.spec` | PyInstaller 构建配置（重新编译时用） |
| `README_duk_fetch.md` | 本说明文档 |

---

## 快速开始

### 双击运行（最简单）

直接双击 `dist/duk_fetch.exe`，使用默认参数（XAUUSD，M30 → M15 → M5）：

```
 duk_fetch.exe
```

运行后会提示确认，输入 `y` 开始拉取。

### 命令行运行

```bash
# 默认参数
duk_fetch.exe

# 指定品种
duk_fetch.exe --symbol EURUSD

# 指定周期（支持 M1/M5/M15/M30/H1/H4/D1/W1）
duk_fetch.exe --timeframes M5 M15 H1

# 指定代理
duk_fetch.exe --proxy http://127.0.0.1:10808

# 忽略断点重新拉取
duk_fetch.exe --no-resume

# 自定义所有路径
duk_fetch.exe --symbol XAUUSD \
              --timeframes M5 M15 M30 \
              --cache-dir ./my_data \
              --checkpoint ./my_ckpt.json \
              --log-file ./my_log.txt \
              --proxy http://127.0.0.1:10808
```

---

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--symbol, -s` | `XAUUSD` | 交易品种 |
| `--timeframes, -t` | `M30 M15 M5` | 要拉取的周期列表（可多个） |
| `--cache-dir, -c` | `data/cache` | 数据缓存目录 |
| `--checkpoint` | `tmp/duk_checkpoint.json` | 断点文件路径 |
| `--log-file, -l` | `tmp/duk_fetch_log.txt` | 日志文件路径 |
| `--proxy, -p` | 自动检测 | HTTP 代理地址（不指定则读取环境变量 `HTTPS_PROXY`/`HTTP_PROXY`/`ALL_PROXY`） |
| `--no-resume` | 关闭 | 忽略已有断点，从头开始拉取 |
| `--save-every` | `50` | 每多少天增量保存一次 parquet |
| `--progress-every` | `10` | 每多少天输出一次进度日志 |
| `--cp-every` | `5` | 每多少天保存一次断点 |

---

## 拉取顺序与数据量参考

默认拉取顺序（从小到大，先快速出结果）：

| 顺序 | 周期 | 预估数据量（20 年） | 预估耗时 |
|------|------|-------------------|---------|
| 1 | M30 | ~53 万根 | ~30 min |
| 2 | M15 | ~77 万根 | ~50 min |
| 3 | M5 | ~170 万根 | ~100 min |

> 实际耗时取决于网络和代理速度，XAUUSD 约 6 小时完成全部周期。

---

## 断点续传

工具支持断点续传，中途中断后重新运行会自动从上次停止的位置继续：

- **断点文件**：`tmp/duk_checkpoint.json`，记录每个周期的当前日期位置
- **增量保存**：每 50 天自动保存一次 parquet，崩溃不丢数据
- **频繁 checkpoint**：每 5 天保存一次断点，防止长时间崩溃丢失进度

如果要从头重新拉取（比如换品种或换周期），加 `--no-resume` 参数。

---

## 输出文件格式

输出为标准 Parquet 文件，列结构如下：

```
time (int64)        - Unix 秒级时间戳（UTC）
open  (float64)     - 开盘价
high  (float64)     - 最高价
low   (float64)     - 最低价
close (float64)     - 收盘价
volume (float64)    - 成交量
tick_volume (float64) - 跳数（= volume，兼容使用）
```

文件名格式：`{SYMBOL}_{TIMEFRAME}_Dukascopy.parquet`

示例：
- `XAUUSD_M30_Dukascopy.parquet`
- `XAUUSD_M15_Dukascopy.parquet`
- `XAUUSD_M5_Dukascopy.parquet`

---

## 依赖

### 编译版 exe

无需安装任何依赖，双击即用。

### 源码版（Python 运行）

```bash
pip install pandas pyarrow
```

Python 版本要求：3.9+

---

## 重新编译 exe

如果需要重新编译（比如修改了代码）：

```bash
# 安装 PyInstaller（如未安装）
pip install pyinstaller

# 编译
pyinstaller duk_fetch_app.spec --noconfirm --clean
```

编译产物在 `dist/duk_fetch.exe`。

---

## 常见用法

### 拉取特定品种

```bash
# 欧元/美元
duk_fetch.exe --symbol EURUSD

# 英镑/美元
duk_fetch.exe --symbol GBPUSD

# 比特币/美元
duk_fetch.exe --symbol BTCUSD
```

### 只拉取特定周期

```bash
# 只拉 5 分钟和 15 分钟
duk_fetch.exe --timeframes M5 M15

# 只拉日线
duk_fetch.exe --timeframes D1

# 拉取所有可用周期
duk_fetch.exe --timeframes M1 M5 M15 M30 H1 H4 D1 W1
```

### 使用代理

```bash
# 指定代理
duk_fetch.exe --proxy http://127.0.0.1:10808

# 通过环境变量
set HTTPS_PROXY=http://127.0.0.1:10808
duk_fetch.exe
```

### 自定义数据目录

```bash
duk_fetch.exe --cache-dir "D:\my_data\XAUUSD"
```

---

## 故障排查

### 连接超时

如果大量出现超时错误，检查：
1. 代理是否正确配置（`--proxy` 参数或环境变量）
2. 网络是否正常
3. 可以尝试降低 `--progress-every` 观察具体哪些日期失败

### 提示 "Dukascopy 主机不可达"

这只是探测警告，不影响实际拉取。工具会继续尝试请求。如果确实无法连接，检查代理设置。

### 数据不完整

- 删除断点文件后用 `--no-resume` 重新拉取
- 检查 `--cache-dir` 指向的目录是否有写入权限

### 内存不足

工具会将数据全量加载到内存后保存。如果内存不足（比如 M5 拉 20 年约 170 万根），可以：
- 减少 `--save-every` 的值（比如改为 20）让 parquet 保存更频繁
- 分批次拉取（比如先拉 M30，完成后再拉 M5）

---

## 数据说明

- 数据来源：Dukascopy 公开历史 API（`jetta.dukascopy.com/v1/candles`）
- 价格类型：BID（买价）
- 时间范围：XAUUSD 约 2006-10-05 至今
- 周末数据自动跳过
- 节假日无交易数据（API 返回空）

---

## 与原脚本的区别

本工具是从项目根目录的 `duk_fetch_all_v3.py` 重构而来，主要变化：

| 特性 | 原脚本 | 本工具 |
|------|--------|--------|
| 依赖 miaosuan 包 | 是（`sys.path.insert`） | 否（完全自包含） |
| 命令行参数 | 无 | 完整 argparse CLI |
| 编译为 exe | 否 | 是（PyInstaller） |
| 品种配置 | 硬编码 XAUUSD | 可配置（默认 XAUUSD） |
| 周期选择 | 固定 M30/M15/M5 | 可自由选择任意组合 |
| 代理配置 | 硬编码 | 可配置 + 环境变量自动检测 |
| 路径配置 | 硬编码绝对路径 | 可配置相对路径 |
