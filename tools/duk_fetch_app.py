#!/usr/bin/env python3
"""
Dukascopy 历史数据分批拉取工具 (Standalone Edition)
=====================================================

完全自包含版本，不依赖 miaosuan 项目代码。

核心流程（v2）:
    1. 按年倒序生成批次: batch0 = [当年-01-01, 今天+1天)，更老批次 =
       [Y-01-01, 年轻相邻批 start + overlap_days)，尾批起点 2006-10-05；
       窗口统一左闭右开 [start, end)。
    2. 双文件: 最终 {sym}_{tf}_Dukascopy.parquet（文件名不变）+
       暂存 {sym}_{tf}_Dukascopy.stage.parquet（仅当前批未合并数据）。
    3. 批次状态机: pending → fetching → fetched → verified → merged；
       周期全部合并后 status=completed。
    4. 合并前强制重叠验证（resume 同样触发）: 在
       [年轻批 start, 本批 end) 内比较 stage 与 final 的公共时间戳
       ΔOHLC、超容差 bar 占比、缺口；不通过则中止、不合并、保留 stage。
    5. 断点 checkpoint v2 顶层带 version=2；旧格式视为无有效断点
       （保留最终文件，从 batch0 重新开始），不崩溃。

用法:
    duk_fetch_app.py                                  # 默认参数运行
    duk_fetch_app.py --symbol XAUUSD                  # 指定品种
    duk_fetch_app.py --timeframes M5 M15              # 只拉取指定周期
    duk_fetch_app.py --no-resume                      # 忽略断点重新拉取
    duk_fetch_app.py --max-batches 1                  # 调试限量
    duk_fetch_app.py --help                           # 查看全部选项
"""

import sys
import os
import time
import json
import calendar
import socket
import urllib.request
import urllib.error
import argparse
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# ── 确保 pandas / pyarrow 可用 ──
try:
    import pandas as pd
except ImportError:
    print("错误: 需要 pandas 库，请运行: pip install pandas pyarrow")
    sys.exit(1)

# ── 全局 socket 超时 ──
SOCKET_TIMEOUT = 60
socket.setdefaulttimeout(SOCKET_TIMEOUT)


# ══════════════════════════════════════════════════════════════════
# 自定义异常
# ══════════════════════════════════════════════════════════════════

class DataError(Exception):
    """数据获取错误"""
    pass


class VerificationError(Exception):
    """重叠验证失败（中止运行、不合并、保留 stage）"""
    pass


# ══════════════════════════════════════════════════════════════════
# 配置常量
# ══════════════════════════════════════════════════════════════════

DUK_BASE = "https://jetta.dukascopy.com/v1/candles"
DUKASCOPY_HOST = "jetta.dukascopy.com"

#: 断点文件版本（v2: 分批 + 重叠验证状态机）
CHECKPOINT_VERSION = 2

#: 输出列（最终文件与 stage 文件共用）
COLUMNS = ("time", "open", "high", "low", "close", "volume", "tick_volume")

#: 妙算周期 -> Dukascopy API 路径段
_TIMEFRAME_MAP = {
    "M1": ("minute", 60),
    "M5": ("minute", 300),
    "M15": ("minute", 900),
    "M30": ("minute", 1800),
    "H1": ("hour", 3600),
    "H4": ("hour", 14400),
    "D1": ("day", 86400),
    "W1": ("day", 604800),
}

#: 默认拉取顺序（从小到大，先快速出结果）
DEFAULT_TIMEFRAMES = ["M30", "M15", "M5"]

#: 批次默认状态
BATCH_STATUSES = ("pending", "fetching", "fetched", "verified", "merged")


# ══════════════════════════════════════════════════════════════════
# 核心工具函数
# ══════════════════════════════════════════════════════════════════

def _symbol_to_duk(symbol: str) -> str:
    """把品种转为 Dukascopy 路径风格（如 XAUUSD -> XAU-USD）"""
    s = (symbol or "").strip().upper()
    if "/" in s:
        return s.replace("/", "-")
    if "-" in s:
        return s
    if len(s) == 6:
        return s[:3] + "-" + s[3:]
    return s


def _build_url(duk_symbol: str, tf_path: str, year: int, month: int | None = None, day: int | None = None) -> str:
    """构建 Dukascopy API URL"""
    price_type = "BID"
    if tf_path == "day":
        return f"{DUK_BASE}/day/{duk_symbol}/{price_type}/{year}"
    elif tf_path == "hour":
        if month is None:
            raise ValueError("hour timeframe requires month")
        return f"{DUK_BASE}/hour/{duk_symbol}/{price_type}/{year}/{month}"
    elif tf_path == "minute":
        if month is None or day is None:
            raise ValueError("minute timeframe requires month and day")
        return f"{DUK_BASE}/minute/{duk_symbol}/{price_type}/{year}/{month}/{day}"
    else:
        raise ValueError(f"Unknown timeframe path: {tf_path}")


def _iso_to_epoch(iso: str) -> int:
    """ISO 日期 (YYYY-MM-DD, UTC) -> Unix 秒"""
    y, m, d = (int(x) for x in iso.split("-"))
    return int(datetime(y, m, d, tzinfo=timezone.utc).timestamp())


def _epoch_to_iso(epoch: int) -> str:
    """Unix 秒 -> ISO 日期字符串 (UTC)"""
    return datetime.fromtimestamp(int(epoch), tz=timezone.utc).strftime("%Y-%m-%d")


def _epoch_to_str(epoch: int) -> str:
    """Unix 秒 -> 可读时间 (UTC)，用于日志样本"""
    return datetime.fromtimestamp(int(epoch), tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def _passes_grid_filter(timestamp_ms: int, target_tf_seconds: int) -> bool:
    """检查 bar 时间戳（毫秒）是否落在目标周期的网格上。

    M1 不做过滤（minute endpoint 原生粒度即 60s，所有 bar 都在网格上）；
    M5/M15/M30/H1/H4 检查分钟/小时能否被周期间隔整除。
    gap-fill 平盘 bar 与真实 bar 共用此函数，统一过滤。
    """
    if target_tf_seconds <= 60:
        return True
    bar_dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
    if target_tf_seconds == 300:  # M5
        return bar_dt.minute % 5 == 0 and bar_dt.second == 0
    elif target_tf_seconds == 900:  # M15
        return bar_dt.minute % 15 == 0 and bar_dt.second == 0
    elif target_tf_seconds == 1800:  # M30
        return bar_dt.minute % 30 == 0 and bar_dt.second == 0
    elif target_tf_seconds == 3600:  # H1
        return bar_dt.minute == 0 and bar_dt.second == 0
    elif target_tf_seconds == 14400:  # H4
        return bar_dt.hour % 4 == 0 and bar_dt.minute == 0 and bar_dt.second == 0
    return True


def _parse_response(data: dict, target_tf_seconds: int) -> list[dict]:
    """解析 Dukascopy API 返回的数据为统一格式"""
    if not data:
        return []

    base_ts = data.get("timestamp", 0)
    multiplier = data.get("multiplier", 1.0)
    shift = data.get("shift", 0)
    times = data.get("times", [])
    opens_delta = data.get("opens", [])
    highs_delta = data.get("highs", [])
    lows_delta = data.get("lows", [])
    closes_delta = data.get("closes", [])
    volumes = data.get("volumes", [])

    if not times:
        return []

    open_units = round(data.get("open", 0) / multiplier) if multiplier else 0
    high_units = round(data.get("high", 0) / multiplier) if multiplier else 0
    low_units = round(data.get("low", 0) / multiplier) if multiplier else 0
    close_units = round(data.get("close", 0) / multiplier) if multiplier else 0

    rows = []
    timestamp_ms = base_ts
    prev_close_units = close_units

    for i, time_delta in enumerate(times):
        gap = time_delta - (0 if i == 0 else 1)
        for g in range(gap):
            flat_ts = timestamp_ms + (g + (0 if i == 0 else 1)) * shift
            # gap-fill 平盘 bar 必须通过周期网格过滤，否则 M5/M15/M30
            # 会产生越界时间戳（例如 M5 网格上的 00:01/00:02/00:03 等分钟）
            if not _passes_grid_filter(flat_ts, target_tf_seconds):
                continue
            flat_price = prev_close_units * multiplier
            rows.append({
                "time": int(flat_ts // 1000),
                "open": float(flat_price),
                "high": float(flat_price),
                "low": float(flat_price),
                "close": float(flat_price),
                "volume": 0.0,
            })

        timestamp_ms += time_delta * shift

        if i < len(opens_delta):
            open_units += opens_delta[i]
        if i < len(highs_delta):
            high_units += highs_delta[i]
        if i < len(lows_delta):
            low_units += lows_delta[i]
        if i < len(closes_delta):
            close_units += closes_delta[i]

        prev_close_units = close_units

        if i < len(opens_delta) and i < len(highs_delta) and i < len(lows_delta) and i < len(closes_delta):
            o = open_units * multiplier
            h = high_units * multiplier
            l = low_units * multiplier
            c = close_units * multiplier
            v = volumes[i] if i < len(volumes) else 0.0

            # 真实 bar 同样走网格过滤（与 gap-fill 统一）
            if not _passes_grid_filter(timestamp_ms, target_tf_seconds):
                continue

            rows.append({
                "time": int(timestamp_ms // 1000),
                "open": float(o),
                "high": float(h),
                "low": float(l),
                "close": float(c),
                "volume": float(v or 0.0),
            })
    return rows


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    """规整 DataFrame 为标准格式并排序去重（单文件自洽场景用 keep='last'）"""
    required = ("time", "open", "high", "low", "close", "volume")
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=[*required, "tick_volume"])

    df = df.copy()
    df["time"] = df["time"].astype("int64")
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype("float64")
    if "tick_volume" not in df.columns:
        df["tick_volume"] = df["volume"]
    df = df.sort_values("time").reset_index(drop=True)
    df = df[~df["time"].duplicated(keep="last")].reset_index(drop=True)
    return df[[*required, "tick_volume"]]


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """统一列类型/列序（合并后使用，不做去重——调用方已按 keep='first' 去重）"""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=[*COLUMNS])
    df = df.copy()
    df["time"] = df["time"].astype("int64")
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype("float64")
    if "tick_volume" not in df.columns:
        df["tick_volume"] = df["volume"]
    else:
        df["tick_volume"] = (
            df["tick_volume"].where(df["tick_volume"].notna(), df["volume"]).astype("float64")
        )
    return df[list(COLUMNS)]


def _probe_host(host: str, port: int = 443, timeout: float = 3.0) -> bool:
    """TCP 连通性探测"""
    try:
        infos = socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_STREAM)
    except Exception:
        return False
    for info in infos[:1]:
        sock = None
        try:
            addr = info[4]
            conn_host = str(addr[0])
            conn_port = int(addr[1])
            sock = socket.create_connection((conn_host, conn_port), timeout=timeout)
            return True
        except Exception:
            continue
        finally:
            if sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass
    return False


# ══════════════════════════════════════════════════════════════════
# Dukascopy 拉取器（自包含）
# ══════════════════════════════════════════════════════════════════

class DukascopyFetcher:
    """Dukascopy 公开历史行情拉取器（自包含，无外部依赖）"""

    def __init__(
        self,
        proxy: str | None = None,
        timeout: int = 60,
        probe_timeout: int = 10,
    ) -> None:
        self._proxy = proxy
        self._timeout = timeout
        self._probe_timeout = probe_timeout
        self._avail_cache: bool | None = None

    def _build_opener(self):
        """构建带代理的 opener"""
        if self._proxy:
            handler = urllib.request.ProxyHandler({"http": self._proxy, "https": self._proxy})
            return urllib.request.build_opener(handler)
        return urllib.request.build_opener()

    def is_available(self) -> bool:
        """主机可达性探测"""
        if self._avail_cache is not None:
            return self._avail_cache
        self._avail_cache = _probe_host(DUKASCOPY_HOST, 443, float(self._probe_timeout))
        return self._avail_cache

    def http_get(self, url: str) -> Any:
        """GET + JSON 解析，带代理和重试"""
        opener = self._build_opener()
        req = urllib.request.Request(url, headers={"User-Agent": "MiaoSuan-Toolkit/1.0"})

        max_retries = 3
        for attempt in range(max_retries):
            try:
                with opener.open(req, timeout=self._timeout) as resp:
                    body = resp.read()
                try:
                    return json.loads(body.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    raise DataError(f"HTTP 返回非 JSON: {url} ({exc})") from exc
            except Exception as exc:
                if attempt == max_retries - 1:
                    raise DataError(f"HTTP GET 失败 (重试 {max_retries} 次后): {url} ({exc})") from exc
                time.sleep(1 * (attempt + 1))

    def fetch_day(self, duk_symbol: str, year: int, month: int, day: int, target_tf_seconds: int) -> list[dict]:
        """拉取单天的分钟线数据"""
        url = _build_url(duk_symbol, "minute", year, month, day)
        data = self.http_get(url)
        return _parse_response(data, target_tf_seconds)

    def fetch_month(self, duk_symbol: str, year: int, month: int, target_tf_seconds: int) -> list[dict]:
        """拉取单月的小时线数据"""
        url = _build_url(duk_symbol, "hour", year, month)
        data = self.http_get(url)
        return _parse_response(data, target_tf_seconds)

    def fetch_year(self, duk_symbol: str, year: int, target_tf_seconds: int) -> list[dict]:
        """拉取单年的日线数据"""
        url = _build_url(duk_symbol, "day", year)
        data = self.http_get(url)
        return _parse_response(data, target_tf_seconds)


# ══════════════════════════════════════════════════════════════════
# 日志
# ══════════════════════════════════════════════════════════════════

class Logger:
    """简单的文件+控制台日志器"""

    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def log(self, msg: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"{ts} {msg}"
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        print(line, flush=True)

    def reset(self):
        """清空日志文件"""
        with open(self.log_file, "w", encoding="utf-8") as f:
            pass


# ══════════════════════════════════════════════════════════════════
# 断点管理（checkpoint v2）
# ══════════════════════════════════════════════════════════════════

def load_checkpoint(checkpoint_path: Path) -> dict:
    """加载断点。

    仅接受 v2 格式（顶层 version=2 的 dict）；旧格式/损坏文件一律
    返回 {} —— 调用方视为「无有效断点」，从 batch0 重新开始（最终
    文件保留），不崩溃。
    """
    if not checkpoint_path.exists():
        return {}
    try:
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return {}
    if not isinstance(data, dict) or data.get("version") != CHECKPOINT_VERSION:
        return {}
    return data


def save_checkpoint(checkpoint_path: Path, cp: dict):
    """保存断点（先写临时文件再原子替换，防止崩溃产生半截 JSON）"""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(cp, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, checkpoint_path)


# ══════════════════════════════════════════════════════════════════
# 批次生成（按年倒序 + 重叠窗口）
# ══════════════════════════════════════════════════════════════════

def build_batches(
    today: date,
    overlap_days: int = 15,
    first_year: int = 2006,
    first_month: int = 10,
    first_day: int = 5,
) -> list[dict]:
    """生成按年倒序的拉取批次，窗口统一 [start, end) 左闭右开。

    - batch0 = [当年-01-01, 今天+1天)（即含今天）；
    - 更老批次 = [Y-01-01, 年轻相邻批 start + overlap_days)，
      因此每批都向已拉取区间延伸 overlap_days 天；
    - 尾批 start 不早于 first_date（XAUUSD 数据起点 2006-10-05）。

    返回 [{"start": date, "end": date}, ...]，index 0 为最新批。
    """
    if isinstance(today, datetime):
        today = today.date()
    first_date = date(first_year, first_month, first_day)

    start0 = date(today.year, 1, 1)
    if start0 < first_date:
        start0 = first_date
    end0 = today + timedelta(days=1)  # end 为右开边界：包含今天
    if end0 <= start0:                # 防御：today 早于数据起点
        end0 = start0 + timedelta(days=1)

    batches: list[dict] = [{"start": start0, "end": end0}]

    year = today.year - 1
    while year >= first_year:
        # 老批次的 end = 年轻相邻批 start + overlap_days（伸进已拉取区间）
        batch_end = batches[-1]["start"] + timedelta(days=overlap_days)
        batch_start = date(year, 1, 1)
        if batch_start < first_date:
            batch_start = first_date
        batches.append({"start": batch_start, "end": batch_end})
        year -= 1

    return batches


def _overlap_start_iso(batches: list[dict], index: int) -> str:
    """当前批的重叠区起点 ISO 日期 = 年轻相邻批 start；batch0 为其自身 start。"""
    if index > 0:
        return batches[index - 1]["start"]
    return batches[index]["start"]


def _serialize_batches(batches: list[dict]) -> list[dict]:
    """把 build_batches 的 date 窗口序列化为 checkpoint 中的批次记录"""
    return [
        {
            "start": b["start"].isoformat(),
            "end": b["end"].isoformat(),
            "status": "pending",
            "cursor": None,
            "days_ok": 0,
            "days_failed": 0,
            "failed_days": [],
        }
        for b in batches
    ]


def build_tf_entry(batches: list[dict]) -> dict:
    """构造某周期的全新断点条目（batch0 起步）"""
    return {
        "status": "pending",
        "batch_index": 0,
        "batches": _serialize_batches(batches),
    }


# ══════════════════════════════════════════════════════════════════
# 抓取项生成（端点粒度 minute/hour/day，窗口内按升序）
# ══════════════════════════════════════════════════════════════════

def _generate_items(tf_path: str, start: date, end: date) -> list[dict]:
    """生成批次窗口 [start, end) 内的抓取项。

    每项: {label, start, end(右开), args}，其中 end 为该项覆盖的
    右开日期边界，用于游标推进与 resume 判定。
    """
    items: list[dict] = []
    one = timedelta(days=1)

    if tf_path == "minute":
        # 按天拉取（跳过周末）
        d = start
        while d < end:
            if d.weekday() < 5:
                items.append({
                    "label": d.isoformat(),
                    "start": d,
                    "end": d + one,
                    "args": (d.year, d.month, d.day),
                })
            d += one
    elif tf_path == "hour":
        # 按月拉取（覆盖与窗口相交的月份）
        y, m = start.year, start.month
        while True:
            cover_start = date(y, m, 1)
            if cover_start >= end:
                break
            cover_end = date(y + 1, 1, 1) if m == 12 else date(y, m + 1, 1)
            items.append({
                "label": f"{y}-{m:02d}",
                "start": cover_start,
                "end": cover_end,
                "args": (y, m),
            })
            y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    elif tf_path == "day":
        # 按年拉取（覆盖与窗口相交的年份）
        for yr in range(start.year, end.year + 1):
            cover_start = date(yr, 1, 1)
            cover_end = date(yr + 1, 1, 1)
            if cover_start < end and cover_end > start:
                items.append({
                    "label": str(yr),
                    "start": cover_start,
                    "end": cover_end,
                    "args": (yr,),
                })
    else:
        raise ValueError(f"Unknown timeframe path: {tf_path}")

    return items


def _fetch_item(
    fetcher: DukascopyFetcher,
    duk_symbol: str,
    tf_path: str,
    item: dict,
    interval_sec: int,
    batch_end_epoch: int,
) -> list[dict]:
    """抓取单项并解析；行按批次右边界截断（hour/day 月份/年份可能越界）"""
    args = item["args"]
    if tf_path == "minute":
        y, m, d = args
        url = _build_url(duk_symbol, "minute", y, m, d)
    elif tf_path == "hour":
        y, m = args
        url = _build_url(duk_symbol, "hour", y, m)
    else:
        url = _build_url(duk_symbol, "day", args[0])

    data = fetcher.http_get(url)
    rows = _parse_response(data, interval_sec)
    if batch_end_epoch is not None:
        rows = [r for r in rows if r["time"] < batch_end_epoch]
    return rows


def _contiguous_cursor(items: list[dict], failed: set[str]) -> date | None:
    """已成功连续前缀的最后一个覆盖日（失败项阻断游标推进）"""
    cursor: date | None = None
    for item in items:
        if item["label"] in failed:
            break
        cursor = item["end"] - timedelta(days=1)
    return cursor


# ══════════════════════════════════════════════════════════════════
# 双文件读写（最终文件 + 暂存文件）
# ══════════════════════════════════════════════════════════════════

def final_parquet_path(cache_dir: Path, symbol: str, tf_name: str) -> Path:
    """最终文件路径（文件名不变）"""
    return cache_dir / f"{symbol}_{tf_name}_Dukascopy.parquet"


def stage_parquet_path(cache_dir: Path, symbol: str, tf_name: str) -> Path:
    """暂存文件路径（仅当前批未合并数据）"""
    return cache_dir / f"{symbol}_{tf_name}_Dukascopy.stage.parquet"


def load_partial_parquet(cache_dir: Path, symbol: str, tf_name: str) -> pd.DataFrame | None:
    """加载最终文件中已有的数据。

    严格区分两种情况（R3）:
      - 文件不存在 → None（真正的首次拉取，verify 可跳过对比）；
      - 文件存在但读取损坏 → 抛 DataError。绝不静默返回 None ——
        否则损坏文件会被 verify 当作「首次拉取」放行，随后被当前批
        stage 覆盖，静默丢弃全部历史数据。
    """
    path = final_parquet_path(cache_dir, symbol, tf_name)
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        raise DataError(
            f"最终文件损坏或无法读取: {path} ({exc})。"
            "已中止且不会覆盖该文件。处理建议: 先备份该文件，再将其移走/删除后重新运行"
            "（会从 batch0 重新拉取），或修复 parquet 文件后重试。"
        ) from exc


def load_final_parquet(cache_dir: Path, symbol: str, tf_name: str) -> pd.DataFrame:
    """加载最终文件（不存在时返回带标准列的空表）"""
    df = load_partial_parquet(cache_dir, symbol, tf_name)
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=[*COLUMNS])
    return df


def load_stage_parquet(stage_path: Path) -> pd.DataFrame:
    """加载暂存文件（不存在时返回带标准列的空表）"""
    if stage_path.exists():
        try:
            df = pd.read_parquet(stage_path)
            if df is not None and len(df) > 0:
                return df
        except Exception:
            pass
    return pd.DataFrame(columns=[*COLUMNS])


def save_stage_parquet(stage_path: Path, rows: list[dict]):
    """写暂存文件（只放当前批未合并数据）"""
    stage_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=[*COLUMNS])
    df = _finalize(df)
    df.to_parquet(stage_path, index=False)


def write_final_parquet(path: Path, df: pd.DataFrame):
    """写最终文件（仅在验证通过后由合并流程调用）。

    原子写（R3）: 先写同目录临时文件再 os.replace —— 崩溃/断电只可能
    产生被丢弃的 tmp 文件，绝不会留下半截的最终文件。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, path)


# ══════════════════════════════════════════════════════════════════
# 重叠验证（合并前强制门禁）
# ══════════════════════════════════════════════════════════════════

def verify_batch_overlap(
    final_df: pd.DataFrame,
    stage_df: pd.DataFrame,
    overlap_start_iso: str,
    batch_end_iso: str,
    interval_sec: int,
    tol_price: float = 0.01,
    tol_mismatch_pct: float = 0.5,
) -> tuple[bool, dict]:
    """在重叠窗口 [overlap_start, batch_end) 内比较 stage 与 final。

    检查项:
      1. 公共时间戳的 ΔOHLC 绝对差 —— 超过 tol_price 的 bar 计为超容差；
      2. 超容差 bar 占比 —— 超过 tol_mismatch_pct(%) 即失败；
      3. 缺口 —— stage 内相邻时间戳间隔 > 1.5×周期间隔、且 final 在该
         缺口内部有数据（证明此处本应有 bar）即计一处缺口，任何缺口都
         视为失败（周末/休市两侧都无数据，不会误报）。

    返回 (ok, report)。final 无重叠数据时视为首次拉取，跳过对比（通过）；
    stage 重叠区无数据视为拉取失败（不通过）。
    """
    os_epoch = _iso_to_epoch(overlap_start_iso)
    be_epoch = _iso_to_epoch(batch_end_iso)
    report: dict = {
        "overlap_start": overlap_start_iso,
        "batch_end": batch_end_iso,
        "stage_rows": 0,
        "final_rows": 0,
        "common": 0,
        "mismatch_pct": 0.0,
        "max_diffs": {},
        "gap_count": 0,
        "samples": [],
        "reasons": [],
        "note": None,
    }

    if stage_df is None or len(stage_df) == 0:
        report["reasons"].append("stage 文件无数据")
        return False, report
    stage_w = stage_df[(stage_df["time"] >= os_epoch) & (stage_df["time"] < be_epoch)]
    report["stage_rows"] = int(len(stage_w))
    if len(stage_w) == 0:
        report["reasons"].append(
            f"重叠区 [{overlap_start_iso}, {batch_end_iso}) 无 stage 数据"
        )
        return False, report

    if final_df is None or len(final_df) == 0:
        report["note"] = "final 尚无数据，跳过重叠对比（首次拉取）"
        return True, report
    final_w = final_df[(final_df["time"] >= os_epoch) & (final_df["time"] < be_epoch)]
    report["final_rows"] = int(len(final_w))
    if len(final_w) == 0:
        report["note"] = "final 在重叠区无数据，跳过重叠对比"
        return True, report

    # ── 1&2: 公共时间戳 ΔOHLC + 超容差占比 ──
    cols = ["open", "high", "low", "close"]
    common = final_w[["time", *cols]].merge(
        stage_w[["time", *cols]], on="time", suffixes=("_f", "_s")
    )
    report["common"] = int(len(common))
    if len(common) == 0:
        report["reasons"].append("重叠区两侧无公共时间戳（覆盖错位）")
        return False, report

    diffs: dict[str, pd.Series] = {
        c: (common[f"{c}_f"] - common[f"{c}_s"]).abs() for c in cols
    }
    report["max_diffs"] = {c: float(diffs[c].max()) for c in cols}

    bad_mask: pd.Series | None = None
    for c in cols:
        over = diffs[c] > tol_price
        bad_mask = over if bad_mask is None else (bad_mask | over)
    mismatch_pct = 100.0 * float(bad_mask.sum()) / len(common)
    report["mismatch_pct"] = mismatch_pct
    if mismatch_pct > tol_mismatch_pct:
        report["reasons"].append(
            f"超容差 bar 占比 {mismatch_pct:.3f}% > {tol_mismatch_pct}%"
        )
        row_max = pd.concat([diffs[c] for c in cols], axis=1).max(axis=1)
        for idx in row_max.nlargest(5).index:
            t = int(common.loc[idx, "time"])
            report["samples"].append(
                f"{_epoch_to_str(t)} "
                f"ΔO={diffs['open'].loc[idx]:.4f} ΔH={diffs['high'].loc[idx]:.4f} "
                f"ΔL={diffs['low'].loc[idx]:.4f} ΔC={diffs['close'].loc[idx]:.4f}"
            )

    # ── 3: 缺口（stage 相邻间隔 > 1.5×周期 且 final 在缺口内有数据） ──
    gap_threshold = 1.5 * float(interval_sec)
    s_times = stage_w.sort_values("time")["time"].reset_index(drop=True)
    f_times = final_w["time"]
    gap_count = 0
    for i in range(len(s_times) - 1):
        t1 = int(s_times.iat[i])
        t2 = int(s_times.iat[i + 1])
        if t2 - t1 <= gap_threshold:
            continue
        has_final_inside = bool(((f_times > t1) & (f_times < t2)).any())
        if has_final_inside:
            gap_count += 1
            if gap_count <= 5:
                report["samples"].append(
                    f"缺口 {_epoch_to_str(t1)} → {_epoch_to_str(t2)} (gap={t2 - t1}s)"
                )
    report["gap_count"] = gap_count
    if gap_count:
        report["reasons"].append(
            f"重叠区缺口 {gap_count} 处 (gap > 1.5×{interval_sec}s)"
        )

    return len(report["reasons"]) == 0, report


# ══════════════════════════════════════════════════════════════════
# 合并（仅 verified 后执行）
# ══════════════════════════════════════════════════════════════════

def merge_batch(final_df: pd.DataFrame, stage_df: pd.DataFrame, overlap_start_iso: str) -> pd.DataFrame:
    """把 stage 数据并入 final，final 在重叠区优胜。

    规则:
      - staging 取 time < overlap_start 的行（新区域）；
      - 加上重叠区中 final 缺失的行（补洞）；
      - concat(final, 子集) → 稳定排序 → 按 time 去重 keep='first'
        （final 优胜）。

    幂等：重复执行结果不变。
    """
    os_epoch = _iso_to_epoch(overlap_start_iso)
    stage = stage_df if stage_df is not None else pd.DataFrame(columns=[*COLUMNS])

    if final_df is None or len(final_df) == 0:
        merged = stage.copy()
    else:
        final_times = set(final_df["time"].tolist())
        mask = (stage["time"] < os_epoch) | (~stage["time"].isin(final_times))
        subset = stage.loc[mask]
        merged = pd.concat([final_df, subset], ignore_index=True)

    merged = merged.sort_values("time", kind="mergesort").reset_index(drop=True)
    merged = merged.loc[~merged["time"].duplicated(keep="first")].reset_index(drop=True)
    return _normalize_df(merged)


# ══════════════════════════════════════════════════════════════════
# 单批抓取（含游标续拉与批末重试）
# ══════════════════════════════════════════════════════════════════

def fetch_batch(
    *,
    fetcher: DukascopyFetcher,
    duk_symbol: str,
    tf_path: str,
    tf_name: str,
    interval_sec: int,
    batch: dict,
    cache_dir: Path,
    symbol: str,
    logger: Logger,
    save_every: int,
    progress_every: int,
    cp_every: int,
    checkpoint_path: Path,
    cp: dict,
    resume_stage: bool,
) -> None:
    """抓取一个批次到 stage 文件，结束时置 batch.status='fetched'。

    - 只生成当前批次窗口内的抓取项（minute 为工作日）；
    - resume_stage=True 时先加载已有 stage（崩溃续拉），游标之前的
      项不重复抓取，上轮 failed_days 项强制重试；
    - 每 save_every 项写 stage+checkpoint，每 cp_every 项写 checkpoint；
    - 批末对 failed_days 做最多 2 轮重试，仍失败的记入报告。
    """
    start_d = date.fromisoformat(batch["start"])
    end_d = date.fromisoformat(batch["end"])
    batch_end_epoch = _iso_to_epoch(batch["end"])
    items = _generate_items(tf_path, start_d, end_d)
    total = len(items)
    one = timedelta(days=1)

    cursor = date.fromisoformat(batch["cursor"]) if batch.get("cursor") else None
    prev_failed = set(batch.get("failed_days") or [])

    stage_path = stage_parquet_path(cache_dir, symbol, tf_name)
    if resume_stage:
        stage_df = load_stage_parquet(stage_path)
        stage_rows = stage_df.to_dict("records") if len(stage_df) else []
        logger.log(f"    续拉: stage 已有 {len(stage_rows)} 根, cursor={batch.get('cursor')}")
    else:
        stage_path.unlink(missing_ok=True)
        stage_rows = []

    failed: set[str] = set()
    processed = 0
    last_save = 0
    last_cp = 0
    last_log = 0
    t_start = time.time()

    def persist() -> None:
        """写 stage + checkpoint"""
        save_stage_parquet(stage_path, stage_rows)
        save_checkpoint(checkpoint_path, cp)

    def advance_cursor(done: list[dict]) -> None:
        """R1: 把批内游标进度写入 batch（随后由 persist 落盘）。

        之前 cursor 只在批末写入，批内 checkpoint 内容空转 —— 批中崩溃
        后 resume 会整批重抓。现在每个游标步进（至少 cp_every 节奏）都
        把 cursor/days_ok/days_failed/failed_days 写进断点。
        注意: 必须先写 stage 再写 checkpoint（persist 的顺序），游标
        永不领先 stage，否则 resume 会跳过 stage 里没有的数据。
        """
        cur = _contiguous_cursor(done, failed)
        if cur is None:
            # 本次已处理项全部失败/尚无成功项 → 保持既有游标
            # （更早的成功前缀仍然有效，不回退）
            cur = cursor
        batch["cursor"] = cur.isoformat() if cur else None
        batch["days_ok"] = sum(1 for it in done if it["label"] not in failed)
        batch["days_failed"] = sum(1 for it in done if it["label"] in failed)
        batch["failed_days"] = sorted(failed)

    def attempt(item: dict) -> None:
        nonlocal processed
        try:
            rows = _fetch_item(fetcher, duk_symbol, tf_path, item, interval_sec, batch_end_epoch)
            stage_rows.extend(rows)
        except Exception as exc:
            failed.add(item["label"])
            if len(failed) <= 30:
                logger.log(f"    {item['label']} FAIL: {str(exc)[:100]}")
        processed += 1

    # ── 初次抓取: cursor 之后的项 + 上轮失败项 ──
    todo: list[dict] = []
    for item in items:
        if item["label"] in prev_failed or cursor is None or item["end"] > cursor + one:
            todo.append(item)

    logger.log(f"    窗口 [{batch['start']}, {batch['end']}) 共 {total} 项, 本次待抓 {len(todo)} 项")

    for idx, item in enumerate(todo, 1):
        attempt(item)
        if processed - last_save >= save_every or processed - last_cp >= cp_every:
            # R1: 游标随 stage+checkpoint 一起落盘（任一节奏到点即推进）
            advance_cursor(todo[:idx])
            persist()
            last_save = processed
            last_cp = processed
        if processed - last_log >= progress_every:
            elapsed = time.time() - t_start
            logger.log(
                f"    进度 {processed}/{len(todo)} 项, 失败 {len(failed)}, "
                f"{elapsed:.0f}s, 当前 {item['label']}"
            )
            last_log = processed

    # ── 批末重试（最多 2 轮），仍失败的记入报告 ──
    for round_no in (1, 2):
        if not failed:
            break
        to_retry = [it for it in items if it["label"] in failed]
        logger.log(f"    重试第 {round_no}/2 轮: {len(to_retry)} 项")
        remaining: set[str] = set()
        for item in to_retry:
            try:
                rows = _fetch_item(fetcher, duk_symbol, tf_path, item, interval_sec, batch_end_epoch)
                stage_rows.extend(rows)
            except Exception as exc:
                remaining.add(item["label"])
                logger.log(f"    {item['label']} 重试失败: {str(exc)[:100]}")
        failed = remaining
        persist()

    # ── 收尾: 游标 / 计数 / 状态 → fetched ──
    cursor_after = _contiguous_cursor(items, failed)
    batch["cursor"] = cursor_after.isoformat() if cursor_after else None
    batch["days_ok"] = total - len(failed)
    batch["days_failed"] = len(failed)
    batch["failed_days"] = sorted(failed)
    batch["status"] = "fetched"
    persist()

    if failed:
        logger.log(
            f"    批拉取完成: {total - len(failed)}/{total} 成功, "
            f"仍有 {len(failed)} 项失败（记入报告）: {sorted(failed)[:20]}"
        )
    else:
        logger.log(f"    批拉取完成: {total} 项全部成功")


# ══════════════════════════════════════════════════════════════════
# 主拉取逻辑（批次状态机）
# ══════════════════════════════════════════════════════════════════

def run_fetch(
    symbol: str,
    timeframes: list[str],
    cache_dir: Path,
    checkpoint_path: Path,
    log_file: Path,
    proxy: str | None,
    save_every: int,
    progress_every: int,
    cp_every: int,
    resume: bool = True,
    overlap_days: int = 15,
    tol_price: float = 0.01,
    tol_mismatch_pct: float = 0.5,
    max_batches: int | None = None,
    today: date | None = None,
) -> dict:
    """执行分批拉取任务。

    状态机（每批次）:
        pending  → 开始抓取
        fetching → 按游标续拉（崩溃恢复）
        fetched  → 强制重叠验证
        verified → 合并进最终文件、删 stage
        merged   → 推进下一批
    任何路径（含 resume 的 fetching/fetched/verified）都必须通过验证
    才能合并；验证失败抛 VerificationError 中止运行、保留 stage。
    """
    if today is None:
        today_d = date.today()
    elif isinstance(today, datetime):
        today_d = today.date()
    else:
        today_d = today

    logger = Logger(log_file)
    if not resume:
        logger.reset()
        if checkpoint_path.exists():
            checkpoint_path.unlink()

    logger.log("=== Dukascopy 分批拉取开始（按年倒序 + 重叠验证 + 断点续传 v2）===")
    logger.log(f"品种: {symbol}")
    logger.log(f"周期: {' → '.join(timeframes)}")
    logger.log(f"缓存目录: {cache_dir}")
    logger.log(f"代理: {proxy or '无'}")
    logger.log(f"断点续传: {'开启' if resume else '关闭'}")
    logger.log(f"批次重叠: {overlap_days} 天, 价格容差: {tol_price}, 超容差占比上限: {tol_mismatch_pct}%")
    logger.log(f"基准日期: {today_d}")
    logger.log(f"开始时间: {datetime.now()}")
    logger.log("")

    # 检查连通性
    fetcher = DukascopyFetcher(proxy=proxy, timeout=60)
    if not fetcher.is_available():
        logger.log("警告: Dukascopy 主机不可达，但将继续尝试（可能只是探测失败）")

    duk_symbol = _symbol_to_duk(symbol)
    logger.log(f"Dukascopy 符号: {duk_symbol}")

    # ── 断点加载（v2 校验；旧格式视为无有效断点，保留最终文件） ──
    cp: dict = {}
    if resume and checkpoint_path.exists():
        cp = load_checkpoint(checkpoint_path)
        if not cp:
            logger.log("断点文件为旧格式或损坏：忽略断点（保留最终文件），从 batch0 重新开始")
    if cp and (cp.get("version") != CHECKPOINT_VERSION or cp.get("symbol") != symbol):
        logger.log(
            f"断点版本/品种不匹配 (version={cp.get('version')}, symbol={cp.get('symbol')})："
            "重新初始化（保留最终文件）"
        )
        cp = {}
    if not cp:
        cp = {"version": CHECKPOINT_VERSION, "overlap_days": overlap_days, "symbol": symbol}
    elif cp.get("overlap_days") != overlap_days:
        logger.log(
            f"警告: 断点记录 overlap_days={cp.get('overlap_days')} ≠ 当前 {overlap_days}，"
            "沿用断点内既有批次窗口"
        )
    save_checkpoint(checkpoint_path, cp)

    batches_spec = build_batches(today_d, overlap_days=overlap_days)
    logger.log(
        f"批次: 共 {len(batches_spec)} 批, 最新 [{batches_spec[0]['start']}, {batches_spec[0]['end']}), "
        f"最老 [{batches_spec[-1]['start']}, {batches_spec[-1]['end']})"
    )
    logger.log("")

    # ── 逐周期处理 ──
    for tf_name in timeframes:
        tf_path, interval_sec = _TIMEFRAME_MAP.get(tf_name, (None, None))
        if tf_path is None:
            logger.log(f"跳过不支持的周期: {tf_name}")
            continue

        entry = cp.get(tf_name)
        if not isinstance(entry, dict) or not entry.get("batches"):
            entry = build_tf_entry(batches_spec)
            cp[tf_name] = entry
        entry.setdefault("status", "pending")
        entry.setdefault("batch_index", 0)

        if entry["status"] == "completed":
            logger.log(f"--- {tf_name}: 已完成，跳过 ---")
            continue

        stage_path = stage_parquet_path(cache_dir, symbol, tf_name)
        if not resume:
            stage_path.unlink(missing_ok=True)

        # ── R3: 最终文件完整性预检 —— 损坏立即中止（给可操作提示），
        # 绝不放行到 verify 当「首次拉取」，更不会被 stage 覆盖 ──
        try:
            load_partial_parquet(cache_dir, symbol, tf_name)
        except DataError as exc:
            logger.log(f"  已中止: {exc}")
            raise

        batches = entry["batches"]
        logger.log(f"--- 拉取 {tf_name} (/{tf_path}/ endpoint, 共 {len(batches)} 批) ---")
        entry["status"] = "in_progress"
        save_checkpoint(checkpoint_path, cp)

        merged_this_run = 0
        while entry["batch_index"] < len(batches):
            if max_batches is not None and merged_this_run >= max_batches:
                logger.log(f"  已达到 --max-batches={max_batches}，停止处理后续批次")
                break

            bi = entry["batch_index"]
            batch = batches[bi]
            prior = batch.get("status", "pending")

            if prior == "merged":
                # 防御性推进（崩溃在写状态之后的场景）
                entry["batch_index"] = bi + 1
                save_checkpoint(checkpoint_path, cp)
                continue

            logger.log(
                f"  批 {bi + 1}/{len(batches)} [{batch['start']}, {batch['end']}) 状态={prior}"
            )

            # ── R2: verified + stage 已删除 → 合并其实已完成，直接推进 ──
            # 旧顺序（写final → 删stage → 置merged → 存断点）在「删stage」与
            # 「存断点」之间崩溃会停在此状态；stage 只可能在 write_final 成功
            # 之后被删除，因此 final 已包含合并结果。若不处理，重叠验证会因
            # stage 无数据永远抛 VerificationError，断点死锁。
            if prior == "verified" and not stage_path.exists():
                logger.log("  检测到 verified 且 stage 已删除（合并已完成），直接推进为 merged")
                batch["status"] = "merged"
                entry["batch_index"] = bi + 1
                merged_this_run += 1
                save_checkpoint(checkpoint_path, cp)
                continue

            # ── 抓取阶段（pending / fetching） ──
            if prior in ("pending", "fetching"):
                batch["status"] = "fetching"
                save_checkpoint(checkpoint_path, cp)
                fetch_batch(
                    fetcher=fetcher,
                    duk_symbol=duk_symbol,
                    tf_path=tf_path,
                    tf_name=tf_name,
                    interval_sec=interval_sec,
                    batch=batch,
                    cache_dir=cache_dir,
                    symbol=symbol,
                    logger=logger,
                    save_every=save_every,
                    progress_every=progress_every,
                    cp_every=cp_every,
                    checkpoint_path=checkpoint_path,
                    cp=cp,
                    resume_stage=(prior == "fetching"),
                )
                prior = "fetched"

            # ── 重叠验证（强制门禁；fetched / verified / resume 均触发） ──
            overlap_start_iso = _overlap_start_iso(batches, bi)
            final_df = load_final_parquet(cache_dir, symbol, tf_name)
            stage_df = load_stage_parquet(stage_path)
            ok, report = verify_batch_overlap(
                final_df,
                stage_df,
                overlap_start_iso,
                batch["end"],
                interval_sec=interval_sec,
                tol_price=tol_price,
                tol_mismatch_pct=tol_mismatch_pct,
            )
            logger.log(
                f"  重叠验证 [{overlap_start_iso}, {batch['end']}): "
                f"公共 {report['common']} 根, 超容差 {report['mismatch_pct']:.3f}%, "
                f"缺口 {report['gap_count']} 处 → {'通过' if ok else '失败'}"
            )
            if report.get("note"):
                logger.log(f"  注意: {report['note']}")
            if not ok:
                for sample in report.get("samples", []):
                    logger.log(f"    差异样本: {sample}")
                logger.log(f"  验证失败原因: {'; '.join(report['reasons'])}")
                logger.log("  已中止: 不合并, stage 保留（排查后重新运行将重新验证）")
                batch["status"] = "fetched"
                entry["status"] = "in_progress"
                save_checkpoint(checkpoint_path, cp)
                raise VerificationError(
                    f"{tf_name} 批 {bi} 重叠验证失败: {'; '.join(report['reasons'])}"
                )

            batch["status"] = "verified"
            save_checkpoint(checkpoint_path, cp)

            # ── 合并（仅 verified 后执行） ──
            merged_df = merge_batch(final_df, stage_df, overlap_start_iso)
            write_final_parquet(final_parquet_path(cache_dir, symbol, tf_name), merged_df)
            batch["status"] = "merged"
            if batch.get("failed_days"):
                logger.log(
                    f"  警告: 该批仍有 {len(batch['failed_days'])} 个失败项（已记入报告）: "
                    f"{batch['failed_days'][:10]}"
                )
            entry["batch_index"] = bi + 1
            merged_this_run += 1
            # R2: 先落盘 merged 断点，再删 stage —— 两步之间崩溃只会留下
            # 孤儿 stage（由下一批 fetch_batch 开头的 unlink 自动清理），
            # 不会再出现「断点停在 verified 而 stage 已删」的验证死锁。
            save_checkpoint(checkpoint_path, cp)
            stage_path.unlink(missing_ok=True)
            logger.log(
                f"  合并完成: 最终 {len(merged_df)} 根 "
                f"(+{len(merged_df) - len(final_df)}), stage 已删除"
            )

        # ── 周期收尾 ──
        if entry["batch_index"] >= len(batches):
            stage_path.unlink(missing_ok=True)  # 全部合并完成 → 清理崩溃残留的孤儿 stage
            entry["status"] = "completed"
            entry["completed_at"] = datetime.now().isoformat()
        else:
            entry["status"] = "in_progress"
        save_checkpoint(checkpoint_path, cp)

        failed_report = [
            (bb["start"], bb["failed_days"]) for bb in batches if bb.get("failed_days")
        ]
        if failed_report:
            logger.log(f"  {tf_name} 失败项汇总:")
            for b_start, f_days in failed_report:
                logger.log(f"    {b_start} 批: {len(f_days)} 项 -> {f_days[:20]}")

        out_path = final_parquet_path(cache_dir, symbol, tf_name)
        if out_path.exists():
            df = pd.read_parquet(out_path)
            if len(df):
                min_dt = datetime.fromtimestamp(int(df["time"].min()), tz=timezone.utc)
                max_dt = datetime.fromtimestamp(int(df["time"].max()), tz=timezone.utc)
                logger.log(
                    f"  {tf_name} 收尾: {len(df)} 根, "
                    f"{min_dt.strftime('%Y-%m-%d')} → {max_dt.strftime('%Y-%m-%d')}, "
                    f"输出: {out_path.name}"
                )
        logger.log("")

    # ── 最终统计 ──
    logger.log("=== 最终统计 ===")
    for tf_name in timeframes:
        path = final_parquet_path(cache_dir, symbol, tf_name)
        if path.exists():
            df = pd.read_parquet(path)
            if len(df):
                min_dt = datetime.fromtimestamp(int(df["time"].min()), tz=timezone.utc).strftime("%Y-%m-%d")
                max_dt = datetime.fromtimestamp(int(df["time"].max()), tz=timezone.utc).strftime("%Y-%m-%d")
            else:
                min_dt = max_dt = "-"
            size = path.stat().st_size / 1e6
            logger.log(f"  {tf_name}: {len(df):>8} 根, {min_dt} → {max_dt}, {size:.2f}MB")
        else:
            logger.log(f"  {tf_name}: 文件不存在")

    logger.log("\n=== 本次运行结束 ===")
    logger.log(f"结束时间: {datetime.now()}")
    return cp


# ══════════════════════════════════════════════════════════════════
# 交易日生成（保留兼容）
# ══════════════════════════════════════════════════════════════════

def generate_days(start_year: int = 2006, start_month: int = 10, start_day: int = 5,
                  end_year: int | None = None, end_month: int = 12) -> list[tuple]:
    """生成工作日列表（跳过周末），保留兼容旧调用"""
    if end_year is None:
        end_year = datetime.now().year

    days = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            if year == end_year and month > end_month:
                break
            if year == start_year and month < start_month:
                continue
            if year == start_year and month == start_month and 1 >= start_day:
                continue

            days_in_month = calendar.monthrange(year, month)[1]
            for day in range(1, days_in_month + 1):
                if year == start_year and month == start_month and day < start_day:
                    continue
                dt = datetime(year, month, day, tzinfo=timezone.utc)
                if dt.weekday() >= 5:
                    continue
                days.append((year, month, day))

    return days


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

def parse_args(argv: list[str] | None = None):
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Dukascopy 历史数据分批拉取工具（按年倒序 + 重叠验证 + 断点续传）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s                                      # 默认参数（XAUUSD, M30/M15/M5）
  %(prog)s --symbol EURUSD                      # 指定品种
  %(prog)s --timeframes M5 M15 H1               # 指定周期
  %(prog)s --no-resume                          # 忽略断点重新拉取
  %(prog)s --proxy http://127.0.0.1:10808       # 指定代理
  %(prog)s --max-batches 1                      # 调试: 只处理 1 个批次
  %(prog)s --overlap-days 15 --tol-price 0.01   # 重叠窗口与验证容差
  %(prog)s --cache-dir ./data                   # 自定义缓存目录
        """,
    )
    parser.add_argument(
        "--symbol", "-s",
        default="XAUUSD",
        help="交易品种 (默认: XAUUSD)",
    )
    parser.add_argument(
        "--timeframes", "-t",
        nargs="+",
        default=DEFAULT_TIMEFRAMES,
        help="要拉取的周期列表 (默认: M30 M15 M5)",
    )
    parser.add_argument(
        "--cache-dir", "-c",
        default="data/cache",
        help="数据缓存目录 (默认: data/cache)",
    )
    parser.add_argument(
        "--checkpoint",
        default="tmp/duk_checkpoint.json",
        help="断点文件路径 (默认: tmp/duk_checkpoint.json)",
    )
    parser.add_argument(
        "--log-file", "-l",
        default="tmp/duk_fetch_log.txt",
        help="日志文件路径 (默认: tmp/duk_fetch_log.txt)",
    )
    parser.add_argument(
        "--proxy", "-p",
        default=None,
        help="HTTP 代理地址 (默认: 无)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="忽略已有断点，从头开始拉取",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=50,
        help="每多少天增量保存一次 stage parquet (默认: 50)",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="每多少天输出一次进度 (默认: 10)",
    )
    parser.add_argument(
        "--cp-every",
        type=int,
        default=5,
        help="每多少天保存一次断点 (默认: 5)",
    )
    parser.add_argument(
        "--overlap-days",
        type=int,
        default=15,
        help="批间重叠天数（合并前重叠验证窗口）(默认: 15)",
    )
    parser.add_argument(
        "--tol-price",
        type=float,
        default=0.01,
        help="重叠验证单 bar OHLC 绝对差容差 (默认: 0.01)",
    )
    parser.add_argument(
        "--tol-mismatch-pct",
        type=float,
        default=0.5,
        help="重叠验证超容差 bar 占比上限 %% (默认: 0.5)",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="每个周期最多处理的批次数，调试用 (默认: 不限)",
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="跳过交互确认，直接开始拉取（非交互/后台运行时必须）",
    )
    return parser.parse_args(argv)


def main():
    """主入口"""
    args = parse_args()

    # 解析路径（相对于脚本目录或当前工作目录）
    base_dir = Path(args.cache_dir).resolve()
    cache_dir = base_dir
    checkpoint_path = Path(args.checkpoint).resolve()
    log_file = Path(args.log_file).resolve()

    # 验证周期
    valid_tfs = set(_TIMEFRAME_MAP.keys())
    invalid_tfs = [tf for tf in args.timeframes if tf not in valid_tfs]
    if invalid_tfs:
        print(f"错误: 不支持的周期: {invalid_tfs}")
        print(f"支持的周期: {', '.join(sorted(valid_tfs))}")
        sys.exit(1)

    # 确定代理（如果未指定，尝试从环境变量读取）
    proxy = args.proxy
    if proxy is None:
        for env_var in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY"):
            env_proxy = os.environ.get(env_var)
            if env_proxy:
                proxy = env_proxy
                break

    print("=" * 60)
    print("  Dukascopy 历史数据分批拉取工具")
    print("=" * 60)
    print(f"  品种:       {args.symbol}")
    print(f"  周期:       {' → '.join(args.timeframes)}")
    print(f"  缓存目录:   {cache_dir}")
    print(f"  断点文件:   {checkpoint_path}")
    print(f"  日志文件:   {log_file}")
    print(f"  代理:       {proxy or '无（使用系统默认）'}")
    print(f"  断点续传:   {'关闭（从头开始）' if args.no_resume else '开启'}")
    print(f"  批次重叠:   {args.overlap_days} 天")
    print(f"  验证容差:   ΔOHLC ≤ {args.tol_price}, 超容差占比 ≤ {args.tol_mismatch_pct}%")
    if args.max_batches is not None:
        print(f"  批次限量:   {args.max_batches}（调试）")
    print("=" * 60)

    if not args.yes:
        confirm = input("\n确认开始拉取? (y/n): ")
        if confirm.lower() not in ("y", "yes"):
            print("已取消。")
            sys.exit(0)

    try:
        run_fetch(
            symbol=args.symbol,
            timeframes=args.timeframes,
            cache_dir=cache_dir,
            checkpoint_path=checkpoint_path,
            log_file=log_file,
            proxy=proxy,
            save_every=args.save_every,
            progress_every=args.progress_every,
            cp_every=args.cp_every,
            resume=not args.no_resume,
            overlap_days=args.overlap_days,
            tol_price=args.tol_price,
            tol_mismatch_pct=args.tol_mismatch_pct,
            max_batches=args.max_batches,
        )
    except VerificationError as exc:
        print(f"\n重叠验证失败，已中止（stage 保留待排查）: {exc}")
        sys.exit(2)
    except DataError as exc:
        print(f"\n数据错误，已中止（不会覆盖最终文件）: {exc}")
        sys.exit(3)


if __name__ == "__main__":
    main()
