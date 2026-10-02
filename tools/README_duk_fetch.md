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
| `../tests/test_duk_fetch_app.py` | 单元测试（合成数据，不发真实网络） |

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
| `--save-every` | `50` | 每多少天增量保存一次 stage parquet |
| `--progress-every` | `10` | 每多少天输出一次进度日志 |
| `--cp-every` | `5` | 每多少天保存一次断点 |
| `--overlap-days` | `15` | 批间重叠天数（合并前重叠验证窗口） |
| `--tol-price` | `0.01` | 重叠验证单 bar OHLC 绝对差容差 |
| `--tol-mismatch-pct` | `0.5` | 重叠验证超容差 bar 占比上限（%） |
| `--max-batches` | 不限 | 每个周期最多处理的批次数（调试限量） |

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

## 拉取流程与断点续传

### 分批策略（按年倒序 + 15 天重叠）

拉取按年倒序分批推进，每批从今天往前推进 1 年，批间重叠 `--overlap-days`（默认 15）天。
窗口统一 **[start, end)** 左闭右开：

| 批次 | 窗口 | 说明 |
|------|------|------|
| batch0 | `[当年-01-01, 今天+1天)` | 含今天，网络波动只影响当前批、可独立重试 |
| batch1 | `[去年-01-01, 今年-01-16)` | 末尾伸进已拉取区间 15 天（重叠区） |
| batchN | `[Y-01-01, 年轻相邻批 start + 15天)` | 每批都与年轻批重叠 15 天 |
| 尾批 | `[2006-10-05, 2007-01-16)` | 数据起点 2006-10-05，不早于该日 |

### 双文件与批次状态机

- **最终文件** `{SYMBOL}_{TIMEFRAME}_Dukascopy.parquet`：唯一对外产物，文件名不变；
- **暂存文件** `{SYMBOL}_{TIMEFRAME}_Dukascopy.stage.parquet`：只放当前批未合并数据，
  合并成功后立即删除。

批次生命周期（断点文件 v2 中每批独立记录）：

| 状态 | 含义 | 下一步动作 |
|------|------|-----------|
| `pending` | 未开始 | 进入抓取 |
| `fetching` | 抓取中（崩溃恢复时从 cursor 续拉） | 按天游标续拉 → 批末 2 轮重试 → `fetched` |
| `fetched` | 抓取完成（可能仍有 failed_days 记录在报告） | **强制重叠验证** |
| `verified` | 重叠验证通过 | 合并进最终文件、删 stage → `merged` |
| `merged` | 已合并 | 推进下一批（`batch_index + 1`） |
| `completed` | 该周期全部批次已合并 | 跳过该周期 |

> 任何路径（含 resume 的 `fetching` / `fetched` / `verified`）都**不得跳过验证直接合并**。
> 唯一例外：状态为 `verified` 且 stage 已删除（旧版崩溃窗口的残留状态）—— stage 只会在写入最终文件
> 成功之后才被删除，因此直接推进为 `merged`，避免「stage 无数据 → 永远验证失败」的死锁。

### 重叠验证与截断合并规则

合并前在重叠窗口 `[年轻批 start, 本批 end)` 内做「叠加对比」，stage 与 final 比较：

1. **公共时间戳 ΔOHLC 绝对差**：单 bar 任一列超过 `--tol-price`（默认 0.01）计为超容差；
2. **超容差 bar 占比**：超过 `--tol-mismatch-pct`（默认 0.5%）即失败；
3. **缺口**：stage 内相邻时间戳间隔 > 1.5×周期间隔、且 final 在缺口内部有数据
   （周末/休市两侧都无数据，不会误报），任何缺口即失败。

- **通过** → 状态 `verified`，执行截断合并：
  - stage 取 `time < overlap_start` 的行（新区域）；
  - 加上重叠区中 final 缺失的行（补洞）；
  - `concat(final, 子集)` → 稳定排序 → 按 time 去重 `keep='first'`（**final 在重叠区优胜**）；
  - 原子写回最终文件（临时文件 + `os.replace`）→ 先落盘 `merged` 断点（`batch_index + 1`）→ 再删除 stage。
    合并幂等，重复执行结果不变；「存断点」与「删 stage」之间崩溃只会留下孤儿 stage，由下一批开头自动清理。
- **失败** → **中止运行、不合并、保留 stage**，日志打印差异样本；重新运行会先重新验证。
- 仅当 final 文件**不存在**时按首次拉取跳过对比放行；final 文件**存在但损坏**会立即中止并给出修复提示，
  **绝不覆盖**（防止历史数据被当前批 stage 静默替换）。

### 断点文件（checkpoint v2）

`tmp/duk_checkpoint.json`，顶层带 `version=2`：

```json
{
  "version": 2,
  "overlap_days": 15,
  "symbol": "XAUUSD",
  "M30": {
    "status": "in_progress",
    "batch_index": 1,
    "batches": [
      {"start": "2026-01-01", "end": "2026-07-10", "status": "merged",
       "cursor": "2026-07-09", "days_ok": 137, "days_failed": 0, "failed_days": []}
    ]
  }
}
```

- 读到**旧格式/损坏断点**时视为无有效断点：**保留最终文件**、从 batch0 重新开始，不崩溃；
- 每 `--save-every` 或 `--cp-every` 天（先到者）把**批内游标进度**（cursor/days_ok/days_failed/failed_days）
  随 stage + checkpoint 一起原子落盘 —— 批中崩溃后 resume 从游标续拉，不会整批重抓；
- 批末对 `failed_days` 做最多 **2 轮重试**，仍失败的写入失败报告与日志（不静默跳过）；
- `--no-resume` 忽略断点并清空 stage，从头开始（最终文件仍保留、合并时 final 优胜）。

### 示例命令

```bash
# 正式全量（本机可用代理）
duk_fetch.exe --proxy http://127.0.0.1:10808

# 调试: 只处理 1 个批次，观察日志
duk_fetch.exe --proxy http://127.0.0.1:10808 --max-batches 1

# 收紧验证容差 / 改变重叠窗口
duk_fetch.exe --overlap-days 15 --tol-price 0.01 --tol-mismatch-pct 0.5

# 验证失败后排查: stage 保留在缓存目录，重新运行会先重新验证再决定是否合并
duk_fetch.exe --proxy http://127.0.0.1:10808

# 运行单元测试（合成数据，不发真实网络）
python -m pytest tests/test_duk_fetch_app.py -q --basetemp .pytest_duk1 -p no:cacheprovider
```

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

同目录下还会出现 `{SYMBOL}_{TIMEFRAME}_Dukascopy.stage.parquet` —— 当前批的暂存文件，
仅在拉取过程中存在，验证通过并合并后自动删除（最终文件由合并流程统一写出）。

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

> 注意：如果是**重叠验证失败（退出码 2）**，「删除断点 + `--no-resume` 重拉」无效，
> 请看下面的「退出码与验证失败（exit 2）」一节。

### 退出码与验证失败（exit 2）

#### 退出码表

| 退出码 | 含义 | 典型提示 |
|--------|------|---------|
| `0` | 成功（或在确认提示处输入 `n` 取消） | `已取消。` / 正常完成 |
| `1` | 参数/环境错误（周期非法等） | `错误: 不支持的周期: [...]` |
| `2` | 重叠验证失败（`VerificationError`） | `重叠验证失败，已中止（stage 保留待排查）: ...` |
| `3` | 最终文件损坏（`DataError`，立即中止、绝不覆盖） | `数据错误，已中止（不会覆盖最终文件）: ...` |

> `3` 的处理步骤见上文「提示『最终文件损坏或无法读取』」。

#### exit 2 场景 a：接缝数据不一致（老批 vs 年轻批在重叠区超容差）

**现象**：日志出现

```
  重叠验证 [2025-12-17, 2026-01-01): 公共 N 根, 超容差 x.xxx%, 缺口 0 处 → 失败
    差异样本: ...
  验证失败原因: ...
  已中止: 不合并, stage 保留（排查后重新运行将重新验证）
```

随后进程以退出码 2 结束；断点停在该批 `fetched`，重新运行会先重新验证再决定是否合并。

**处置**：

1. 人工核对日志里的差异样本——差异可能来自上游修改历史价格、混入了换数据源的数据等，先判断哪一侧数据可信；
2. 若确认**新抓数据可信**，临时放宽容差重跑：

   ```bash
   duk_fetch.exe --tol-price 0.05 --tol-mismatch-pct 2.0
   ```

3. 若最终文件一侧整体不可信，则**备份后移走最终文件**（`{SYMBOL}_{TF}_Dukascopy.parquet`），重新全量拉取——final 不存在时按首次拉取放行。

#### exit 2 场景 b：batch0 全窗对比撞门禁

**何时发生**：使用 `--no-resume`，或读到旧格式/损坏断点被当作「无有效断点」而从 batch0 重新开始时，会对**现存最终文件**做整窗比对——`_overlap_start_iso` 对 index 0 返回该批自身的 start，即 **batch0 的验证窗 = 自身全窗**。

**含义**：现存最终文件与当前数据源**整体**不一致（换了数据源、上游修订了历史价等），不是局部接缝问题。

**处置**（与场景 a 相同，二选一）：

- 放宽容差接受差异：调大 `--tol-price` / `--tol-mismatch-pct` 重跑；
- 或**备份并移走最终文件**，再全量重新拉取。

> **重要：此场景下「删除断点 + `--no-resume`」无效。** 重抓得到的数据与源一致，
> 而合并遵循 **final 优胜原则**（重叠区以现存最终文件为准），对比结果不会被翻转，
> 重跑仍会被同一门禁拒掉，永远卡在 exit 2。必须走上面两条路之一。

#### 数据安全保证

**验证失败绝不会覆盖或删除最终文件与 stage**：exit 2 只中止运行，最终文件、stage、断点
全部原样保留，数据始终安全，可按上述步骤人工处置。

### 提示「最终文件损坏或无法读取」

最终 parquet 无法解析时工具会**立即中止**（不会覆盖该文件，也不会按首次拉取放行）。处理步骤：

1. 先备份该文件（保留证据）；
2. 将其移走/删除后重新运行（会从 batch0 重新拉取），或用 parquet 工具修复后重试；
3. 最终文件写入已改为「临时文件 + 原子替换」，正常运行不会再产生半截文件。

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
