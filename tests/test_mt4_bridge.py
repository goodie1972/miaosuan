# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for full license text.

"""MT4 Bridge 数据源测试（``data/fetchers/mt4_bridge``）。

设计目标：
* 用**本地 loopback 假 EA 服务器**（:class:`FakeMT4EA`）严格复刻 FreeMT4Bridge V3
  的线协议（``#`` 分字段、``!`` 结束、bar 为 ``time$open$high$low$close$volume``、
  末尾补一个空字段），因此**不发任何真实网络请求、不依赖本机是否装了 MT4**，
  在 CI 与离线机器上结果一致可复现；
* 协议解析必须按 EA 源码实测格式验证——历史上这里踩过两个坑：
  把 payload 当二维数组解析导致恒返回空；以及 offset 估算在「有休市断线的行情」
  上静默漏数据。两者都在本文件有定向断言；
* ``time_base`` 换算、增量边界、错误路径（连不上 / EA 报错 / 空数据 / 非法周期）
  逐个覆盖，任何一处静默降级都会立刻炸测试。
"""

from __future__ import annotations

import contextlib
import socket
import threading
import time
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from miaosuan.config import (
    DEFAULT_MT4_BRIDGE_HOST,
    DEFAULT_MT4_BRIDGE_PORT,
    DEFAULT_MT4_TIME_BASE,
    data_acquisition_config,
)
from miaosuan.data.acquisition import (
    SOURCE_TYPE_MT4,
    DataAcquisition,
    TypedNetworkSource,
)
from miaosuan.data.fetchers import MT4BridgeFetcher, all_network_sources
from miaosuan.data.fetchers.mt4_bridge import (
    DEFAULT_MT4_PORT,
    MT4_TIMEFRAME,
    MT4BridgeClient,
)
from miaosuan.errors import DataError

# 经纪商时间相对 UTC 的偏移（小时），与真实 Hantec Markets V 实测一致
BROKER_OFFSET_H: float = 3.0
# 测试用的 H1 周期秒数
PERIOD_S: int = 3600
# 标准输出列（与 BaseFetcher._finalize 契约一致）
REQUIRED_COLUMNS: tuple[str, ...] = (
    "time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "tick_volume",
)
# 客户端测试参数：drain/poll 都压到极小，避免每个用例白等
FAST_KWARGS: dict[str, float] = {"timeout": 2.0, "poll_wait": 0.02, "drain_wait": 0.0}

# 反向表：EA 的请求里携带的是 MT4 整数周期（H1 -> 60），不是字符串 "H1"
# —— 历史上这里把 bars 用字符串键存，导致请求的 "60" 永远查不到，恒返回空页。
TF_NUM_TO_NAME: dict[int, str] = {v: k for k, v in MT4_TIMEFRAME.items()}

# 假报价（bid < ask，符合真实经纪商语义）
BID: float = 2100.5
ASK: float = 2100.6


# ── 装置：本地 loopback 假 EA 服务器 ────────────────────────────────────────


class FakeMT4EA:
    """FreeMT4Bridge V3 协议的最小可运行桩。

    严格复刻真实 EA 的线格式（``FreeMT4Bridge.mq4`` 源码实测）：

    * 请求：``F042#-#<symbol>#<tf>#<offset>#<count>!``
    * 响应：``F042#OK#<bar>#<bar>##!``  —— 每根 bar 为
      ``time$open$high$low$close$volume``，**结尾额外补一个空字段**；
    * ``F020``：``F020#OK#<serverTime>#<bid>#<ask>#<spread>#!``。

    bar 数据按 **offset 0 = 最新** 的时间降序存放（与 MT4 ``iTime(i)`` 语义一致），
    请求按 ``[offset, offset+count)`` 切片返回。

    Attributes:
        f042_calls: 收到的 ``F042`` 请求次数（用于验证增量取数没有全量翻页）。
        f042_requests: 每次请求的 ``(offset, count)``，便于断言翻页行为。
    """

    def __init__(
        self,
        bars: dict[tuple[str, str], list[tuple[int, float, float, float, float, float]]]
        | None = None,
        broker_offset_hours: float = BROKER_OFFSET_H,
        fail_all: bool = False,
        f042_error: str | None = None,
    ) -> None:
        self.bars = bars or {}
        self.broker_offset_hours = broker_offset_hours
        self.fail_all = fail_all
        self.f042_error = f042_error
        self.f042_calls = 0
        self.f042_requests: list[tuple[int, int]] = []
        self._srv: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # ── 生命周期 ────────────────────────────────────────────────────────

    @property
    def port(self) -> int:
        assert self._srv is not None
        return self._srv.getsockname()[1]

    def start(self) -> FakeMT4EA:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", 0))
        srv.listen(16)
        srv.settimeout(0.2)
        self._srv = srv
        self._thread = threading.Thread(target=self._run, daemon=True, name="fake-mt4-ea")
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._srv is not None:
            with contextlib.suppress(OSError):
                self._srv.close()
        if self._thread is not None:
            self._thread.join(timeout=3)

    def _run(self) -> None:
        while not self._stop.is_set() and self._srv is not None:
            try:
                conn, _ = self._srv.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            threading.Thread(target=self._handle_one, args=(conn,), daemon=True).start()

    # ── 协议处理 ────────────────────────────────────────────────────────

    def _handle_one(self, conn: socket.socket) -> None:
        """处理一条 TCP 连接上的**多条**顺序请求。

        真实 EA 是长连接 + 轮询模型：客户端连一次后在同一连接上反复收发
        （一次取数 = 多次 F042 翻页）。若这里处理一条就 close，则第二个请求起
        全部 ``ConnectionAbortedError``——历史上正是这样把整套测试打挂的。
        """
        try:
            conn.settimeout(0.3)
            buf = b""
            while True:
                idx = buf.find(b"!")
                if idx >= 0:  # 完整请求已到手，先派完再等下一段
                    req = buf[:idx].decode("utf-8", "replace").split("#")
                    buf = buf[idx + 1 :]
                    conn.sendall((self._dispatch(req) + "!").encode("utf-8"))
                    continue
                try:
                    chunk = conn.recv(4096)
                except TimeoutError:
                    return  # 客户端空闲走人
                except OSError:
                    return
                if not chunk:
                    return  # 客户端主动关闭
                buf += chunk
        finally:
            with contextlib.suppress(OSError):
                conn.close()

    def _dispatch(self, req: list[str]) -> str:
        if not req:
            return "F000#OK#"
        code = req[0]
        if self.fail_all:
            return "F000#ERROR#bridge down#"
        if code == "F000":
            return "F000#OK#"
        if code == "F020":
            server_t = int(time.time() + self.broker_offset_hours * 3600)
            # 逐字复刻 EA 源码第 231 行：
            #   StringFormat("F020#OK#%d#%.5f#%.5f#%.5f#0#", (int)t, ask, bid, bid)
            # 注意最后一个数值位 EA 填的是 **bid** 而非点差（免费 EA 的实现缺陷），
            # 客户端如实透传，此处必须保持一致才能验证解析位置正确。
            return f"F020#OK#{server_t}#{ASK:.5f}#{BID:.5f}#{BID:.5f}#0#"
        if code == "F042":
            self.f042_calls += 1
            if len(req) < 6:
                return "F042#ERROR#bad params#"
            symbol = req[2]
            try:
                tf_num, offset, count = int(req[3]), int(req[4]), int(req[5])
            except ValueError:
                return "F042#ERROR#bad params#"
            # EA 用 (int)StringToInteger(parts[3]) 取整数周期再喂给 iTime()，
            # 故这里必须按整数反查品种键，用字符串 "H1" 查会永远落空。
            timeframe = TF_NUM_TO_NAME.get(tf_num)
            if timeframe is None:
                return "F042#ERROR#bad params#"
            self.f042_requests.append((offset, count))
            if self.f042_error is not None:
                return f"F042#{self.f042_error}#boom#"
            count = max(1, min(count, 5000))
            series = self.bars.get((symbol, timeframe), [])
            page = series[offset : offset + count]
            body = "".join(
                f"{b[0]}${b[1]:.5f}${b[2]:.5f}${b[3]:.5f}${b[4]:.5f}${int(b[5])}#" for b in page
            )
            return f"F042#OK#{body}#"
        return f"{code}#OK#"


def make_h1_bars(
    n: int = 250,
    broker_offset_hours: float = BROKER_OFFSET_H,
    gap_after: int = 120,
    gap_seconds: int = 7200,
) -> list[tuple[int, float, float, float, float, float]]:
    """生成 ``n`` 根 H1 bar，**时间降序**（index 0 = 最新，对应 EA offset 0）。

    OHLC 由 bar 序号派生，便于对拍：第 ``i`` 根（降序）的
    ``open=1000+i, high=open+10, low=open-10, close=open+5, volume=open``。

    在降序第 ``gap_after`` 根之后插入一次 ``gap_seconds`` 休市断线，
    用来验证取数逻辑在**非连续时间轴**上仍然正确。
    """
    now_broker = int(time.time() + broker_offset_hours * 3600)
    now_broker -= now_broker % PERIOD_S  # 对齐到整点
    times: list[int] = []
    t = now_broker
    for i in range(n):
        times.append(t)
        t -= PERIOD_S
        if i == gap_after:
            t -= gap_seconds
    bars: list[tuple[int, float, float, float, float, float]] = []
    for i, ts in enumerate(times):
        v = 1000.0 + i
        bars.append((ts, v, v + 10.0, v - 10.0, v + 5.0, v))
    return bars


@pytest.fixture
def ea() -> FakeMT4EA:
    """默认假 EA：250 根 H1 bar，挂在随机 loopback 端口。"""
    srv = FakeMT4EA(bars={("XAUUSD", "H1"): make_h1_bars(250)}).start()
    try:
        yield srv
    finally:
        srv.stop()


def fetcher(port: int, **overrides: Any) -> MT4BridgeFetcher:
    """构造指向 ``port`` 的快速测试 fetcher。"""
    kwargs: dict[str, Any] = dict(FAST_KWARGS, host="127.0.0.1", port=port)
    kwargs.update(overrides)
    return MT4BridgeFetcher(**kwargs)


# ── 协议层 ───────────────────────────────────────────────────────────────


def test_ping_roundtrip(ea: FakeMT4EA) -> None:
    with MT4BridgeClient("127.0.0.1", ea.port, **FAST_KWARGS) as c:
        assert c.ping() == "OK"


def test_server_time_and_offset_detection(ea: FakeMT4EA) -> None:
    with MT4BridgeClient("127.0.0.1", ea.port, **FAST_KWARGS) as c:
        server_t = c.server_time("XAUUSD")
        assert 0 < (server_t - int(time.time())) / 3600 < 3.05
        assert c.detect_utc_offset_hours() == pytest.approx(BROKER_OFFSET_H, abs=0.26)


def test_quote_parses_bid_ask_spread(ea: FakeMT4EA) -> None:
    """解析位置必须与 EA 源码第 231 行对齐：``(t, ask, bid, bid, 0)``。

    最后一个数值位 EA 填的是 bid 而非点差，客户端照实透传——这里断言的就是
    这个「保真」行为，若将来有人顺手「修正」成真实点差，此测试会同步失败提醒。
    """
    with MT4BridgeClient("127.0.0.1", ea.port, **FAST_KWARGS) as c:
        q = c.quote("XAUUSD")
    assert set(q) == {"time", "ask", "bid", "spread"}
    assert isinstance(q["time"], int)
    assert q["ask"] == ASK
    assert q["bid"] == BID
    assert q["spread"] == BID  # EA 缺陷位：填的是 bid


def test_bars_payload_format_parsed_not_empty(ea: FakeMT4EA) -> None:
    """回归：payload 是 `t$o$h$l$c$v` 扁平串 + 末尾空字段，不是二维数组。

    旧版误按二维数组解析，导致任何请求都恒返回 ``[]``（且被吞成「无数据」）。
    """
    with MT4BridgeClient("127.0.0.1", ea.port, **FAST_KWARGS) as c:
        got = c.bars("XAUUSD", "H1", 0, 5)
    assert len(got) == 5
    first = got[0]
    assert set(first) == {"time", "open", "high", "low", "close", "volume"}
    assert isinstance(first["time"], int)
    assert first["open"] == 1000.0  # 降序 index 0 = 最新
    assert first["close"] == 1005.0
    # offset 0 = 最新，故返回序列应为时间降序
    assert all(got[i]["time"] > got[i + 1]["time"] for i in range(len(got) - 1))


def test_bars_offset_slice(ea: FakeMT4EA) -> None:
    with MT4BridgeClient("127.0.0.1", ea.port, **FAST_KWARGS) as c:
        assert c.bars("XAUUSD", "H1", 7, 3)[0]["open"] == 1007.0
    # 超出历史边界时 EA 返回空页（iTime() == 0），是空列表而非异常
    with MT4BridgeClient("127.0.0.1", ea.port, **FAST_KWARGS) as c:
        assert c.bars("XAUUSD", "H1", 10_000, 10) == []


def test_ea_error_response_raises() -> None:
    """EA 回 ERROR 时必须抛 DataError（默认夹具是正常服务端，需自建故障端）。"""
    srv = FakeMT4EA(bars={("XAUUSD", "H1"): make_h1_bars(50)}, f042_error="ERROR").start()
    try:
        with (
            MT4BridgeClient("127.0.0.1", srv.port, **FAST_KWARGS) as c,
            pytest.raises(DataError, match="F042"),
        ):
            c.bars("XAUUSD", "H1", 0, 5)
    finally:
        srv.stop()


def test_connect_failure_raises_dataerror() -> None:
    with pytest.raises(DataError, match="MT4 Bridge"):
        MT4BridgeClient("127.0.0.1", 1, **FAST_KWARGS).connect()


# ── fetch_full ───────────────────────────────────────────────────────────


def test_fetch_full_contract(ea: FakeMT4EA) -> None:
    df = fetcher(ea.port, page_bars=60, utc_offset_hours=BROKER_OFFSET_H).fetch_full("XAUUSD", "H1")
    assert list(df.columns) == list(REQUIRED_COLUMNS)
    assert len(df) == 250
    assert df["time"].dtype == "int64"
    # offset 0 = 最新，聚合后必须转为**时间升序**
    assert df["time"].is_monotonic_increasing
    assert not df["time"].duplicated().any()
    # OHLC 对拍：升序第 j 行 = 降序第 (249-j) 根
    for j in (0, 1, 124, 248, 249):
        src = 249 - j
        assert df["open"].iloc[j] == 1000.0 + src
        assert df["high"].iloc[j] == 1000.0 + src + 10
        assert df["low"].iloc[j] == 1000.0 + src - 10
        assert df["close"].iloc[j] == 1000.0 + src + 5
        assert df["volume"].iloc[j] == 1000.0 + src
    # tick_volume 与 volume 同值（MT4 的 volume 就是 tick volume）
    assert (df["volume"] == df["tick_volume"]).all()


def test_fetch_full_handles_session_gap(ea: FakeMT4EA) -> None:
    """休市断线不得被填平、跳过或报错。"""
    df = fetcher(ea.port, page_bars=60, utc_offset_hours=BROKER_OFFSET_H).fetch_full("XAUUSD", "H1")
    diffs = df["time"].diff().dropna()
    assert (diffs == PERIOD_S).sum() == 248
    big = diffs[diffs > PERIOD_S]
    assert len(big) == 1
    assert int(big.iloc[0]) == PERIOD_S + 7200


def test_fetch_full_page_boundary_not_losing_last_bar(ea: FakeMT4EA) -> None:
    """回归：分页边界处最后一根 bar 不得因 off-by-one 丢失。"""
    bars = make_h1_bars(121)
    srv = FakeMT4EA(bars={("XAUUSD", "H1"): bars}).start()
    try:
        df = fetcher(srv.port, page_bars=60, utc_offset_hours=BROKER_OFFSET_H).fetch_full(
            "XAUUSD", "H1"
        )
    finally:
        srv.stop()
    assert len(df) == 121
    # 最老的一根（降序 index 120）必须仍在
    assert df["open"].iloc[0] == 1000.0 + 120


def test_fetch_full_respects_max_bars(ea: FakeMT4EA) -> None:
    df = fetcher(ea.port, page_bars=60, max_bars=121, utc_offset_hours=BROKER_OFFSET_H).fetch_full(
        "XAUUSD", "H1"
    )
    assert len(df) == 121
    # 截断后仍保留**最新**的 121 根（因为从 offset 0 向前翻）
    assert df["close"].iloc[-1] == 1005.0


def test_fetch_full_timeframe_case_insensitive(ea: FakeMT4EA) -> None:
    df = fetcher(ea.port, utc_offset_hours=BROKER_OFFSET_H).fetch_full("XAUUSD", "h1")
    assert len(df) == 250


def test_fetch_full_maps_timeframe_to_mt4_constant(ea: FakeMT4EA) -> None:
    assert MT4_TIMEFRAME["H1"] == 60
    with MT4BridgeClient("127.0.0.1", ea.port, **FAST_KWARGS) as c:
        c.bars("XAUUSD", "M15", 0, 1)
    assert ea.f042_requests[-1] == (0, 1)
    # 请求里携带的是 MT4 整数周期 15，而非字符串 "M15"
    with MT4BridgeClient("127.0.0.1", ea.port, **FAST_KWARGS) as c:
        c.bars("XAUUSD", "M15", 0, 2)
    assert ea.f042_requests[-1] == (0, 2)


# ── 错误路径 ─────────────────────────────────────────────────────────────


def test_fetch_full_invalid_timeframe_raises(ea: FakeMT4EA) -> None:
    with pytest.raises(DataError, match="周期"):
        fetcher(ea.port).fetch_full("XAUUSD", "M7")


@pytest.mark.parametrize("bad", ["", "   ", "#X", "X$AU", "X!AU", "NOPE#123"])
def test_bad_symbol_raises(ea: FakeMT4EA, bad: str) -> None:
    with pytest.raises(DataError):
        fetcher(ea.port).fetch_full(bad, "H1")


def test_unknown_symbol_raises_dataerror(ea: FakeMT4EA) -> None:
    """本机无该品种历史时 EA 返回空 → 必须如实报 DataError，不得静默返回空表。"""
    with pytest.raises(DataError, match="无数据"):
        fetcher(ea.port).fetch_full("XAGUSD", "H1")


def test_is_available_true_when_ea_up(ea: FakeMT4EA) -> None:
    assert fetcher(ea.port).is_available() is True


def test_is_available_false_when_ea_down() -> None:
    """连不上不得抛异常，只返回 False（否则会把整个来源列表搞挂）。"""
    assert fetcher(1, **FAST_KWARGS).is_available() is False


def test_is_available_caches_result(ea: FakeMT4EA) -> None:
    f = fetcher(ea.port)
    assert f.is_available() is True
    ea.stop()  # 中途挂掉
    assert f.is_available() is True  # 命中缓存，不再重连


def test_invalid_time_base_raises() -> None:
    with pytest.raises(DataError, match="time_base"):
        MT4BridgeFetcher(time_base="martian")


def test_ea_down_raises_on_fetch(ea: FakeMT4EA) -> None:
    f = fetcher(ea.port)
    ea.stop()
    with pytest.raises(DataError):
        f.fetch_full("XAUUSD", "H1")


# ── 时间基准换算 ─────────────────────────────────────────────────────────


def test_time_base_broker_returns_raw_broker_time(ea: FakeMT4EA) -> None:
    """time_base='broker' 必须逐字保留经纪商时间，不做任何偏移。"""
    df = fetcher(ea.port, time_base="broker", page_bars=60).fetch_full("XAUUSD", "H1")
    newest = int(df["time"].max())
    assert 0 < (newest - int(time.time())) / 3600 < 3.05  # broker ≈ UTC+3


def test_time_base_utc_subtracts_broker_offset(ea: FakeMT4EA) -> None:
    """time_base='utc'（默认）须把经纪商时间减去实测偏移。"""
    now = int(time.time())
    df = fetcher(ea.port, utc_offset_hours=BROKER_OFFSET_H).fetch_full("XAUUSD", "H1")
    newest = int(df["time"].max())
    assert newest <= now + PERIOD_S
    assert newest >= now - 2 * PERIOD_S


def test_time_base_shanghai_is_utc_plus_8(ea: FakeMT4EA) -> None:
    df_utc = fetcher(ea.port, time_base="utc", utc_offset_hours=BROKER_OFFSET_H).fetch_full(
        "XAUUSD", "H1"
    )
    df_sh = fetcher(ea.port, time_base="shanghai", utc_offset_hours=BROKER_OFFSET_H).fetch_full(
        "XAUUSD", "H1"
    )
    delta = df_sh["time"].values - df_utc["time"].values
    assert (delta == 8 * 3600).all()
    # OHLC 与时间无关，不得被换算破坏
    for col in ("open", "high", "low", "close", "volume"):
        assert (df_utc[col] == df_sh[col]).all()


def test_offset_detected_from_f020_when_not_configured(ea: FakeMT4EA) -> None:
    """不显式传 utc_offset_hours 时，必须实测并经 F020 对表。"""
    df = fetcher(ea.port).fetch_full("XAUUSD", "H1")
    assert int(df["time"].max()) <= int(time.time()) + PERIOD_S


def test_describe_reports_host_port_and_base(ea: FakeMT4EA) -> None:
    f = fetcher(ea.port, time_base="shanghai", utc_offset_hours=3.5)
    d = f.describe()
    assert "127.0.0.1" in d and str(ea.port) in d
    assert "shanghai" in d and "+3.5h" in d


# ── 增量取数（历史坑位重点）──────────────────────────────────────────────


def test_incremental_returns_only_bars_after_since(ea: FakeMT4EA) -> None:
    full = fetcher(ea.port, page_bars=60, utc_offset_hours=BROKER_OFFSET_H).fetch_full(
        "XAUUSD", "H1"
    )
    n = len(full)
    since = int(full["time"].iloc[n - 50])  # 只取最后 49 根
    ea.f042_calls = 0
    inc = fetcher(ea.port, page_bars=60, utc_offset_hours=BROKER_OFFSET_H).fetch_incremental(
        "XAUUSD", "H1", since
    )
    assert len(inc) == 49
    assert int(inc["time"].min()) > since
    assert int(inc["time"].max()) == int(full["time"].max())
    assert list(inc.columns) == list(REQUIRED_COLUMNS)


def test_incremental_does_not_full_page(ea: FakeMT4EA) -> None:
    """回归：offset 不能按「墙上小时数」估算——休市使时长 > 根数，
    估算偏大会空转、偏小会**静默漏掉最新数据**。改为翻页时逐页比对时间边界。
    """
    full = fetcher(ea.port, page_bars=60, utc_offset_hours=BROKER_OFFSET_H).fetch_full(
        "XAUUSD", "H1"
    )
    since = int(full["time"].iloc[-50])
    ea.f042_calls = 0
    fetcher(ea.port, page_bars=60, utc_offset_hours=BROKER_OFFSET_H).fetch_incremental(
        "XAUUSD", "H1", since
    )
    # 1 次 head 探测 + 1 页即越界停止；全量则需要 5 页
    assert ea.f042_calls == 2
    assert 5 * 60 > 50  # 全量确实需要 5 页


def test_incremental_reaches_back_across_session_gap(ea: FakeMT4EA) -> None:
    """since 落在休市断线另一侧时，仍须正确穿越断线取全。"""
    full = fetcher(ea.port, page_bars=60, utc_offset_hours=BROKER_OFFSET_H).fetch_full(
        "XAUUSD", "H1"
    )
    # 断线位于升序 index 121 处，取断线两侧各 20 根做边界
    since = int(full["time"].iloc[141])
    inc = fetcher(ea.port, page_bars=60, utc_offset_hours=BROKER_OFFSET_H).fetch_incremental(
        "XAUUSD", "H1", since
    )
    expect = full[full["time"] > since]
    # 用 .tolist() 而非直接比较 Series：expect 继承 full 的索引（142..249），
    # inc 是 reset 后的 0..N-1，直接 == 会按索引对齐成全 NaN 而假失败。
    assert list(inc["time"]) == list(expect["time"])
    assert inc["close"].tolist() == expect["close"].tolist()


def test_incremental_since_older_than_history_returns_everything(ea: FakeMT4EA) -> None:
    inc = fetcher(ea.port, utc_offset_hours=BROKER_OFFSET_H).fetch_incremental("XAUUSD", "H1", 0)
    assert len(inc) == 250


def test_incremental_since_newer_than_history_returns_empty(ea: FakeMT4EA) -> None:
    """没有新数据时返回**空 DataFrame**（带标准列），而不是报错。"""
    inc = fetcher(ea.port, utc_offset_hours=BROKER_OFFSET_H).fetch_incremental(
        "XAUUSD", "H1", int(time.time()) + 30 * 86400
    )
    assert isinstance(inc, pd.DataFrame)
    assert len(inc) == 0
    assert list(inc.columns) == list(REQUIRED_COLUMNS)


def test_incremental_validates_inputs(ea: FakeMT4EA) -> None:
    with pytest.raises(DataError):
        fetcher(ea.port).fetch_incremental("", "H1", 0)
    with pytest.raises(DataError):
        fetcher(ea.port).fetch_incremental("XAUUSD", "M7", 0)


# ── 配置与注册 ───────────────────────────────────────────────────────────


def test_config_defaults() -> None:
    cfg = data_acquisition_config({})
    assert cfg.mt4_bridge_host == DEFAULT_MT4_BRIDGE_HOST == "127.0.0.1"
    assert cfg.mt4_bridge_port == DEFAULT_MT4_BRIDGE_PORT == DEFAULT_MT4_PORT == 23232
    assert cfg.mt4_time_base == DEFAULT_MT4_TIME_BASE == "utc"


def test_config_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MIAOSUAN_MT4_BRIDGE_HOST", "  0.0.0.0  ")
    monkeypatch.setenv("MIAOSUAN_MT4_BRIDGE_PORT", "23333")
    monkeypatch.setenv("MIAOSUAN_MT4_TIME_BASE", "shanghai")
    cfg = data_acquisition_config()
    assert cfg.mt4_bridge_host == "0.0.0.0"  # 已 strip
    assert cfg.mt4_bridge_port == 23333
    assert cfg.mt4_time_base == "shanghai"


def test_config_bad_port_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MIAOSUAN_MT4_BRIDGE_PORT", "not-a-number")
    assert data_acquisition_config().mt4_bridge_port == DEFAULT_MT4_BRIDGE_PORT
    monkeypatch.setenv("MIAOSUAN_MT4_BRIDGE_PORT", "-1")
    assert data_acquisition_config().mt4_bridge_port == DEFAULT_MT4_BRIDGE_PORT


def test_all_network_sources_excludes_mt4_when_host_blank() -> None:
    kinds = {type(s).__name__ for s in all_network_sources(mt4_bridge_host="")}
    assert "MT4BridgeFetcher" not in kinds


def test_all_network_sources_includes_mt4_when_host_set() -> None:
    sources = all_network_sources(mt4_bridge_host="127.0.0.1", mt4_bridge_port=43210)
    mt4 = [s for s in sources if isinstance(s, MT4BridgeFetcher)]
    assert len(mt4) == 1
    assert str(mt4[0].describe()).startswith("MT4 Bridge")


def test_data_acquisition_registers_mt4_source(
    ea: FakeMT4EA, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """构造期即可把 MT4 来源按「数据源类型」登记进来源表。"""
    monkeypatch.setenv("MIAOSUAN_DATA_CACHE_DIR", str(tmp_path))
    acq = DataAcquisition(
        sources=[TypedNetworkSource(fetcher(ea.port, utc_offset_hours=BROKER_OFFSET_H))]
    )
    types = [s["source_type"] for s in acq.list_sources()]
    assert SOURCE_TYPE_MT4 in types


def test_data_acquisition_fetch_from_mt4_end_to_end(
    ea: FakeMT4EA, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """端到端：DataAcquisition → MT4 来源 → 落盘 parquet → 读回校验。"""
    monkeypatch.setenv("MIAOSUAN_DATA_CACHE_DIR", str(tmp_path))
    acq = DataAcquisition(
        sources=[TypedNetworkSource(fetcher(ea.port, utc_offset_hours=BROKER_OFFSET_H))]
    )
    path = acq.fetch("XAUUSD", "H1", source=SOURCE_TYPE_MT4, note="unit-test")
    try:
        back = pd.read_parquet(path)
        assert len(back) == 250
        assert list(back.columns) == list(REQUIRED_COLUMNS)
        assert back["time"].is_monotonic_increasing
        assert not back["time"].duplicated().any()
    finally:
        with contextlib.suppress(OSError):
            path.unlink()
