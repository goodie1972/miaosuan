#!/usr/bin/env python3
"""
Dukascopy 历史数据全周期拉取工具 (Standalone Edition)
=====================================================

完全自包含版本，不依赖 miaosuan 项目代码。
支持断点续传、增量保存、代理、命令行参数配置。

用法:
    duk_fetch.exe                                    # 默认参数运行
    duk_fetch.exe --symbol XAUUSD                    # 指定品种
    duk_fetch.exe --timeframes M5 M15                # 只拉取指定周期
    duk_fetch.exe --no-resume                        # 忽略断点重新拉取
    duk_fetch.exe --help                             # 查看全部选项
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
from datetime import datetime, timezone
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


# ══════════════════════════════════════════════════════════════════
# 配置常量
# ══════════════════════════════════════════════════════════════════

DUK_BASE = "https://jetta.dukascopy.com/v1/candles"
DUKASCOPY_HOST = "jetta.dukascopy.com"

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

            if target_tf_seconds > 60:
                bar_dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
                if target_tf_seconds == 300:  # M5
                    if bar_dt.minute % 5 != 0 or bar_dt.second != 0:
                        continue
                elif target_tf_seconds == 900:  # M15
                    if bar_dt.minute % 15 != 0 or bar_dt.second != 0:
                        continue
                elif target_tf_seconds == 1800:  # M30
                    if bar_dt.minute % 30 != 0 or bar_dt.second != 0:
                        continue
                elif target_tf_seconds == 3600:  # H1
                    if bar_dt.minute != 0 or bar_dt.second != 0:
                        continue
                elif target_tf_seconds == 14400:  # H4
                    if bar_dt.hour % 4 != 0 or bar_dt.minute != 0 or bar_dt.second != 0:
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
    """规整 DataFrame 为标准格式并排序去重"""
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
# 断点管理
# ══════════════════════════════════════════════════════════════════

def load_checkpoint(checkpoint_path: Path) -> dict:
    """加载断点"""
    if checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_checkpoint(checkpoint_path: Path, cp: dict):
    """保存断点"""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(cp, f, ensure_ascii=False, indent=2)


# ══════════════════════════════════════════════════════════════════
# 交易日生成
# ══════════════════════════════════════════════════════════════════

def generate_days(start_year: int = 2006, start_month: int = 10, start_day: int = 5,
                  end_year: int | None = None, end_month: int = 12) -> list[tuple]:
    """生成工作日列表（跳过周末）"""
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
# 增量保存
# ══════════════════════════════════════════════════════════════════

def load_partial_parquet(cache_dir: Path, symbol: str, tf_name: str) -> pd.DataFrame | None:
    """加载已有的部分数据"""
    path = cache_dir / f"{symbol}_{tf_name}_Dukascopy.parquet"
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception:
            pass
    return None


def save_partial_parquet(cache_dir: Path, symbol: str, tf_name: str, rows: list[dict]):
    """增量保存数据"""
    if not rows:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{symbol}_{tf_name}_Dukascopy.parquet"
    df = pd.DataFrame(rows)
    df = _finalize(df)
    df.to_parquet(path, index=False)


# ══════════════════════════════════════════════════════════════════
# 主拉取逻辑
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
):
    """执行拉取任务"""
    logger = Logger(log_file)
    if not resume:
        logger.reset()
        if checkpoint_path.exists():
            checkpoint_path.unlink()

    logger.log("=== Dukascopy 全周期数据拉取开始 ===")
    logger.log(f"品种: {symbol}")
    logger.log(f"周期: {' → '.join(timeframes)}")
    logger.log(f"缓存目录: {cache_dir}")
    logger.log(f"代理: {proxy or '无'}")
    logger.log(f"断点续传: {'开启' if resume else '关闭'}")
    logger.log(f"开始时间: {datetime.now()}")
    logger.log("")

    # 检查连通性
    fetcher = DukascopyFetcher(proxy=proxy, timeout=60)
    if not fetcher.is_available():
        logger.log("警告: Dukascopy 主机不可达，但将继续尝试（可能只是探测失败）")
        logger.log("")

    duk_symbol = _symbol_to_duk(symbol)
    logger.log(f"Dukascopy 符号: {duk_symbol}")

    cp = load_checkpoint(checkpoint_path) if resume else {}
    all_days = generate_days()
    logger.log(f"总工作日: {len(all_days)} 天")
    logger.log("")

    # ── 逐周期拉取 ──
    for tf_name in timeframes:
        tf_path, interval_sec = _TIMEFRAME_MAP.get(tf_name, (None, None))
        if tf_path is None:
            logger.log(f"跳过不支持的周期: {tf_name}")
            continue

        cp_entry = cp.get(tf_name, {"status": "pending"})

        if resume and cp_entry.get("status") == "completed":
            logger.log(f"--- {tf_name}: 已完成，跳过 ---")
            continue

        logger.log(f"--- 拉取 {tf_name} (Dukascopy /{tf_path}/ endpoint) ---")
        cp_entry["status"] = "in_progress"
        save_checkpoint(checkpoint_path, cp)

        # 断点恢复
        start_date = None
        existing_rows = []
        if resume and "year" in cp_entry and "month" in cp_entry and "day" in cp_entry:
            start_date = (cp_entry["year"], cp_entry["month"], cp_entry["day"])
            logger.log(f"  从断点恢复: {start_date[0]}-{start_date[1]:02d}-{start_date[2]:02d}")

        # 加载已保存的部分数据
        partial = load_partial_parquet(cache_dir, symbol, tf_name)
        if partial is not None:
            existing_rows = partial.to_dict("records")
            logger.log(f"  已加载 {len(existing_rows)} 根已保存数据")

        # 生成待拉取的日期
        if tf_path == "minute":
            # 按日拉取
            days = generate_days()
            if start_date:
                sy, sm, sd = start_date
                start_dt = datetime(sy, sm, sd, tzinfo=timezone.utc)
                days = [(y, m, d) for y, m, d in days
                        if datetime(y, m, d, tzinfo=timezone.utc) > start_dt]
        elif tf_path == "hour":
            # 按月拉取
            months = []
            for year in range(2006, datetime.now().year + 1):
                for month in range(1, 13):
                    if year == 2026 and month > 9:
                        break
                    if year == 2006 and month < 10:
                        continue
                    months.append((year, month))
            if start_date:
                sy, sm, _ = start_date
                months = [(y, m) for y, m in months
                          if (y, m) > (sy, sm)]
            days = months  # 复用变量名
        else:
            # 按年拉取（日线）
            years = list(range(2006, datetime.now().year + 1))
            if start_date:
                years = [y for y in years if y > start_date[0]]
            days = [(y, 1, 1) for y in years]  # 复用变量名

        total_items = len(days)
        day_count = cp_entry.get("days_done", 0)
        day_fail = cp_entry.get("days_failed", 0)
        day_ok = cp_entry.get("days_ok", 0)
        last_log_count = day_count
        last_save_count = day_count
        t_start = time.time()

        for i, item in enumerate(days):
            if tf_path == "minute":
                year, month, day = item
                url = _build_url(duk_symbol, "minute", year, month, day)
                try:
                    data = fetcher.http_get(url)
                    rows = _parse_response(data, interval_sec)
                    existing_rows.extend(rows)
                    day_count += 1
                    day_ok += 1
                except Exception as e:
                    day_fail += 1
                    if day_fail <= 30:
                        logger.log(f"  {year}-{month:02d}-{day:02d} FAIL: {str(e)[:100]}")
            elif tf_path == "hour":
                year, month = item
                url = _build_url(duk_symbol, "hour", year, month)
                try:
                    data = fetcher.http_get(url)
                    rows = _parse_response(data, interval_sec)
                    existing_rows.extend(rows)
                    day_count += 1
                    day_ok += 1
                except Exception as e:
                    day_fail += 1
                    if day_fail <= 30:
                        logger.log(f"  {year}-{month:02d} FAIL: {str(e)[:100]}")
            else:
                year = item[0]
                url = _build_url(duk_symbol, "day", year)
                try:
                    data = fetcher.http_get(url)
                    rows = _parse_response(data, interval_sec)
                    existing_rows.extend(rows)
                    day_count += 1
                    day_ok += 1
                except Exception as e:
                    day_fail += 1
                    if day_fail <= 30:
                        logger.log(f"  {year} FAIL: {str(e)[:100]}")

            # 更新 checkpoint
            if tf_path == "minute":
                cp_entry["year"] = item[0]
                cp_entry["month"] = item[1]
                cp_entry["day"] = item[2]
                cp_entry["current_date"] = f"{item[0]}-{item[1]:02d}-{item[2]:02d}"
            elif tf_path == "hour":
                cp_entry["year"] = item[0]
                cp_entry["month"] = item[1]
                cp_entry["current_date"] = f"{item[0]}-{item[1]:02d}"
            else:
                cp_entry["year"] = item[0]
                cp_entry["current_date"] = f"{item[0]}"

            cp_entry["days_done"] = day_count
            cp_entry["days_failed"] = day_fail
            cp_entry["days_ok"] = day_ok
            cp_entry["last_update"] = datetime.now().isoformat()
            cp_entry["bars_count"] = len(existing_rows)

            # 进度日志
            if day_count - last_log_count >= progress_every:
                elapsed = time.time() - t_start
                total_processed = day_count + day_fail
                rate = total_processed / elapsed if elapsed > 0 else 0
                remaining_items = total_items - (i + 1)
                remaining_sec = remaining_items / rate if rate > 0 else 0
                current_date = cp_entry.get("current_date", "?")
                logger.log(
                    f"  {tf_name} 进度: {day_ok}成功/{day_fail}失败/{day_count}有效, "
                    f"{day_count}/{total_items} ({day_count/total_items*100:.0f}%), "
                    f"{elapsed:.0f}s, ~{rate:.1f}/s, "
                    f"当前: {current_date}, "
                    f"预计剩余 {remaining_sec/60:.0f}min"
                )
                last_log_count = day_count

            # 保存 checkpoint
            if (i + 1) % cp_every == 0:
                save_checkpoint(checkpoint_path, cp)

            # 增量保存 parquet
            if day_count - last_save_count >= save_every:
                save_partial_parquet(cache_dir, symbol, tf_name, existing_rows)
                save_checkpoint(checkpoint_path, cp)
                last_save_count = day_count

        # 保存最终数据
        elapsed = time.time() - t_start
        save_partial_parquet(cache_dir, symbol, tf_name, existing_rows)
        cp_entry["status"] = "completed"
        cp_entry["total_bars"] = len(existing_rows)
        cp_entry["completed_at"] = datetime.now().isoformat()
        save_checkpoint(checkpoint_path, cp)

        # 输出结果
        out_path = cache_dir / f"{symbol}_{tf_name}_Dukascopy.parquet"
        if out_path.exists():
            df = pd.read_parquet(out_path)
            min_dt = datetime.fromtimestamp(int(df["time"].min()), tz=timezone.utc)
            max_dt = datetime.fromtimestamp(int(df["time"].max()), tz=timezone.utc)
            logger.log(
                f"  {tf_name} 完成: {len(df)} 根, "
                f"{min_dt.strftime('%Y-%m-%d')} → {max_dt.strftime('%Y-%m-%d')}, "
                f"耗时 {elapsed/60:.1f}min, 输出: {out_path.name}"
            )
        else:
            logger.log(f"  {tf_name} 完成但无输出文件")

        logger.log("")

    # ── 最终统计 ──
    logger.log("=== 最终统计 ===")
    for tf_name in timeframes:
        path = cache_dir / f"{symbol}_{tf_name}_Dukascopy.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            min_dt = datetime.fromtimestamp(int(df["time"].min()), tz=timezone.utc).strftime("%Y-%m-%d")
            max_dt = datetime.fromtimestamp(int(df["time"].max()), tz=timezone.utc).strftime("%Y-%m-%d")
            size = path.stat().st_size / 1e6
            logger.log(f"  {tf_name}: {len(df):>8} 根, {min_dt} → {max_dt}, {size:.2f}MB")
        else:
            logger.log(f"  {tf_name}: 文件不存在")

    logger.log(f"\n=== 全部完成 ===")
    logger.log(f"结束时间: {datetime.now()}")


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Dukascopy 历史数据全周期拉取工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s                                      # 默认参数（XAUUSD, M30/M15/M5）
  %(prog)s --symbol EURUSD                      # 指定品种
  %(prog)s --timeframes M5 M15 H1               # 指定周期
  %(prog)s --no-resume                          # 忽略断点重新拉取
  %(prog)s --proxy http://127.0.0.1:10808       # 指定代理
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
        help="每多少天增量保存一次 parquet (默认: 50)",
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
    return parser.parse_args()


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
    print("  Dukascopy 历史数据全周期拉取工具")
    print("=" * 60)
    print(f"  品种:       {args.symbol}")
    print(f"  周期:       {' → '.join(args.timeframes)}")
    print(f"  缓存目录:   {cache_dir}")
    print(f"  断点文件:   {checkpoint_path}")
    print(f"  日志文件:   {log_file}")
    print(f"  代理:       {proxy or '无（使用系统默认）'}")
    print(f"  断点续传:   {'关闭（从头开始）' if args.no_resume else '开启'}")
    print("=" * 60)

    confirm = input("\n确认开始拉取? (y/n): ")
    if confirm.lower() not in ("y", "yes"):
        print("已取消。")
        sys.exit(0)

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
    )


if __name__ == "__main__":
    main()
