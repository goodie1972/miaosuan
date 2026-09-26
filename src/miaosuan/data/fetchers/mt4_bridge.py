# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for full license text.

"""MT4 Bridge 数据源（本机 MT4 终端挂 ``FreeMT4Bridge`` EA 后，经 localhost TCP 读行情）。

与其余 fetcher 的区别：它**不需要外网**——数据来自本机 MT4 终端已缓存/已下载的
历史 K 线，MT4 自己从经纪商服务器取。因此在本机断网时仍能取回历史数据（实测
断网环境 ``F042`` 正常返回，仅实时报价 ``bid/ask`` 为 0）。

协议（逐条对照 EA 源码 ``FreeMT4Bridge.mq4`` 核对，非推测）::

    请求  "F042#-#XAUUSD#60#0#500!"          # '#' 分字段，'!' 结束一条消息
    响应  "F042#OK#t$o$h$l$c$v#t$o$h$l$c$v##!"
                              └─ 每根 bar: time$open$high$low$close$volume
    状态  "F042#ERROR#bad params#!"          # 参数不足等错误

EA 用 ``OnTimer`` 每秒轮询一次连接，故**每次请求后必须等满一拍再收**，否则会
拿到空读——这是本实现里唯一反直觉之处。

时间戳口径（重要）::

    ``MT4 iTime()`` 返回**经纪商服务器时间**，非 UTC。实测 Hantec Markets 为
    UTC+3（用 ``F020`` 的 ``TimeCurrent`` 与本机 UTC 对表得出，见
    :meth:`MT4BridgeClient.detect_utc_offset_hours`）。

    :attr:`time_base` 控制输出口径，默认 ``"utc"`` 以符合
    ``panel.py`` 的「时间统一 UTC epoch 秒」契约。注意 ``D:\\K线数据`` 下的
    既有缓存实测为 **UTC+8**（与 MT4 broker 时间差 +5h，用 OHLC 中位偏差
    0.25 vs 其他偏移 5~18 判定），与契约不符——故本实现默认**不**沿用旧缓存
    口径，需对齐时显式传 ``time_base="shanghai"``。
"""

from __future__ import annotations

import socket
import time
from typing import Any

import pandas as pd

from ...errors import DataError
from .base import BaseFetcher

__all__ = ["MT4BridgeClient", "MT4BridgeFetcher", "DEFAULT_MT4_PORT"]


def _settings_mt4_port() -> int:
    """从统一设置系统读取 MT4 Bridge 端口（settings.yaml 可改，支持热重载）。

    延迟导入避免 import 期副作用；设置系统不可用时回落 23232。
    """
    try:
        from ...settings import get_config
        return get_config().mt4.port
    except Exception:  # pragma: no cover - 防御性
        return 23232


#: FreeMT4Bridge V3 默认监听端口（EA 源码 `#property` 未固定，社区默认 23232）。
#: 注意：这是**回退值**；实际运行时默认端口由统一设置系统决定
#: （:func:`_settings_mt4_port`，对应 ``settings.yaml → mt4.port``）。
DEFAULT_MT4_PORT: int = 23232

#: 妙算周期标识 -> MQL4 ``PERIOD_*`` 常数（EA 按整数接收，见源码第 236 行）。
MT4_TIMEFRAME: dict[str, int] = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 60,
    "H4": 240,
    "D1": 1440,
}

#: ``time_base`` 可选值 -> 说明。
_TIME_BASES = ("broker", "utc", "shanghai")

#: 输出时间基准偏移（相对 UTC 的小时数）。
_TZ_OFFSET_HOURS: dict[str, float] = {"utc": 0.0, "shanghai": 8.0}

#: EA 单页最多返回根数（源码第 239 行：`if(count > 5000) count = 5000`）。
_EA_MAX_COUNT: int = 5000

#: bar 字段数：time$open$high$low$close$volume
_BAR_FIELDS: int = 6


class MT4BridgeClient:
    """FreeMT4Bridge V3 的极简 TCP 客户端（仅取行情，不做交易）。

    每次 :meth:`connect` 建立一条独立连接；调用方负责 :meth:`close`（或用 ``with``）。
    刻意**不复用连接做长轮询**：EA 的响应与请求不是一对一的强绑定，连接久了会
    读到上一轮的残留消息，短连接更稳。

    Args:
        host: 监听地址，默认 ``127.0.0.1``。
        port: 监听端口，默认从统一设置系统读取（settings.yaml → mt4.port）。
        timeout: 单次 socket 操作超时（秒）。
        poll_wait: 发出请求后等待 EA 轮询一拍的时间（秒）。EA 约 1s 一次，
            取值小于 1 会偶发空读；过大则白白变慢。
        drain_wait: 发送前清空残留响应时单次 ``recv`` 的超时（秒）。仅用于
            吸收上一轮残留，取 0 即可让测试快速跑完。
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int | None = None,
        timeout: float = 3.0,
        poll_wait: float = 1.6,
        drain_wait: float = 0.25,
    ) -> None:
        self.host = host
        self.port = port if port is not None else _settings_mt4_port()
        self.timeout = timeout
        self.poll_wait = poll_wait
        self.drain_wait = drain_wait
        self._sock: socket.socket | None = None

    # ── 连接管理 ─────────────────────────────────────────────────────────

    def connect(self) -> MT4BridgeClient:
        """建立 TCP 连接；失败抛出 :class:`DataError`（不静默）。"""
        try:
            sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        except OSError as exc:
            raise DataError(
                f"无法连接 MT4 Bridge {self.host}:{self.port}: {exc}",
                context={
                    "host": self.host,
                    "port": self.port,
                    "hint": "确认 MT4 终端已启动、FreeMT4Bridge EA 已挂到图表上，"
                    "且 EA 属性里已启用「允许 WebRequest」/网络访问",
                },
            ) from exc
        self._sock = sock
        return self

    def close(self) -> None:
        """关闭连接（幂等）。"""
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None

    def __enter__(self) -> MT4BridgeClient:
        return self.connect()

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ── 协议核心 ─────────────────────────────────────────────────────────

    @property
    def sock(self) -> socket.socket:
        if self._sock is None:
            raise DataError("MT4 Bridge 未连接：先调用 connect()")
        return self._sock

    def _drain(self, wait: float | None = None) -> None:
        """清空可能残留的上一条响应，避免错位读取。"""
        assert self._sock is not None
        self._sock.settimeout(self.drain_wait if wait is None else wait)
        try:
            while True:
                chunk = self._sock.recv(65536)
                if not chunk:
                    break
        except (TimeoutError, OSError):
            pass
        finally:
            self._sock.settimeout(self.timeout)

    def _recv_message(self) -> str:
        """收一条以 ``!`` 结束的消息，返回去掉结尾 ``!`` 的正文。"""
        buf = b""
        while b"!" not in buf:
            try:
                chunk = self.sock.recv(65536)
            except TimeoutError as exc:
                raise DataError(
                    f"等待 MT4 Bridge 响应超时（已收 {len(buf)} 字节）",
                    context={"host": self.host, "port": self.port, "received": len(buf)},
                ) from exc
            except OSError as exc:
                raise DataError(f"MT4 Bridge 连接中断: {exc}") from exc
            if not chunk:
                raise DataError(f"MT4 Bridge 对端关闭连接（已收 {len(buf)} 字节）") from None
            buf += chunk
        return buf[: buf.index(b"!")].decode("utf-8", "replace")

    def call(self, *fields: str, wait: float | None = None) -> list[str]:
        """发一条命令并按 ``#`` 切分返回。

        Args:
            *fields: 命令字段，如 ``"F042", "-", "XAUUSD", "60", "0", "500"``。
            wait: 等待 EA 轮询的时间；``None`` 用实例默认 :attr:`poll_wait`。

        Returns:
            响应字段列表，如 ``["F042", "OK", "123$..$..", "", ...]``。

        Raises:
            DataError: 连接/超时错误。
        """
        assert self._sock is not None
        payload = ("#".join(fields) + "!").encode("utf-8")
        try:
            self._drain()
            self.sock.sendall(payload)
            time.sleep(wait if wait is not None else self.poll_wait)
            return self._recv_message().split("#")
        except DataError:
            raise
        except OSError as exc:
            raise DataError(f"MT4 Bridge 请求失败 ({self.host}:{self.port}): {exc}") from exc

    # ── 高层命令 ─────────────────────────────────────────────────────────

    def ping(self, wait: float | None = None) -> str:
        """``F000`` 探活，成功返回 ``"OK"``。"""
        parts = self.call("F000", "-", wait=wait)
        if len(parts) < 2 or parts[1] != "OK":
            raise DataError(
                f"MT4 Bridge 探活失败: 响应 {parts[:3]}",
                context={"host": self.host, "port": self.port},
            )
        return parts[1]

    def server_time(self, symbol: str = "-", wait: float | None = None) -> int:
        """返回经纪商服务器时间（Unix 秒），来自 ``F020`` 的 ``TimeCurrent()``。

        ⚠ 这是**经纪商时间**而非 UTC；与本机 UTC 的差即经纪商时区偏移。
        市场未接入时返回的仍是服务器时间（实测断网下正常），故可用于对表。
        """
        parts = self.call("F020", "-", symbol, wait=wait)
        if len(parts) < 3 or parts[1] != "OK":
            raise DataError(
                f"F020 失败: {parts[:4]}", context={"host": self.host, "port": self.port}
            )
        try:
            return int(parts[2])
        except ValueError as exc:
            raise DataError(f"F020 返回的时间无法解析: {parts[2]!r}") from exc

    def detect_utc_offset_hours(self, wait: float | None = None) -> float:
        """实测经纪商时间相对 UTC 的偏移（小时），四舍五入到 0.5h。

        经纪商用 EET 类时区时夏季 +3 / 冬季 +2，故必须实测而非硬编码。
        """
        delta = self.server_time(wait=wait) - int(time.time())
        return round(delta / 3600.0 * 2) / 2

    def bars(
        self,
        symbol: str,
        timeframe: str,
        offset: int = 0,
        count: int = 500,
        wait: float | None = None,
    ) -> list[dict[str, Any]]:
        """取一段历史 K 线（``F042``）。

        ``offset`` 从当前未完成 bar 开始向后数（0 = 最新）。返回按 **offset 升序**
        即**时间降序**排列，调用方需自行排序。

        Args:
            symbol: MT4 品种名（如 ``XAUUSD``）。名称不匹配时 EA 的 ``iTime()``
                立即返回 0 → 本方法返回空列表（**不是**异常，见下方说明）。
            timeframe: 妙算周期标识（``M1``/``M5``/…/``D1``）。
            offset: 起始 offset。
            count: 请求根数（EA 上限 5000，超出自动裁剪）。

        Returns:
            ``[{"time","open","high","low","close","volume"}, ...]``；时间已转为
            **整数 Unix 秒（经纪商时间口径）**。

        Raises:
            DataError: 周期不支持、EA 报错、或协议解析异常。
        """
        tf = MT4_TIMEFRAME.get(timeframe.upper())
        if tf is None:
            raise DataError(
                f"MT4 不支持的周期: {timeframe}",
                context={"supported": sorted(MT4_TIMEFRAME)},
            )
        count = max(1, min(int(count), _EA_MAX_COUNT))
        parts = self.call("F042", "-", symbol, str(tf), str(int(offset)), str(count), wait=wait)
        if len(parts) < 3:
            raise DataError(f"F042 响应字段不足: {parts[:4]}", context={"symbol": symbol})
        if parts[1] != "OK":
            raise DataError(
                f"MT4 取数失败: {symbol} {timeframe} → {parts[1]} {parts[2] if len(parts) > 2 else ''}",
                context={"symbol": symbol, "timeframe": timeframe, "response": parts[:4]},
            )
        out: list[dict[str, Any]] = []
        for cell in parts[2:]:  # 末尾有一个空串（EA 结尾补了 '#'）
            if not cell:
                continue
            f = cell.split("$")
            if len(f) < _BAR_FIELDS:
                continue  # 容忍末尾不完整的半根
            try:
                out.append(
                    {
                        "time": int(f[0]),
                        "open": float(f[1]),
                        "high": float(f[2]),
                        "low": float(f[3]),
                        "close": float(f[4]),
                        "volume": float(f[5]),
                    }
                )
            except ValueError as exc:
                raise DataError(f"F042 返回了非法 OHLC 字段: {cell!r}") from exc
        return out

    def quote(self, symbol: str, wait: float | None = None) -> dict[str, float]:
        """取实时报价（``F020``）：``{"time","ask","bid","spread"}``。

        ⚠ 经纪商未接入实时行情时 ``bid/ask`` 为 ``0.0``（历史 K 线仍正常）。
        返回 0 不等于出错，调用方需自行判断。
        """
        parts = self.call("F020", "-", symbol, wait=wait)
        if len(parts) < 6 or parts[1] != "OK":
            raise DataError(f"F020 失败: {parts[:5]}")
        return {
            "time": int(parts[2]),
            "ask": float(parts[3]),
            "bid": float(parts[4]),
            "spread": float(parts[5]),
        }


class MT4BridgeFetcher(BaseFetcher):
    """MT4 终端历史 K 线读取器（本机 EA 桥接，**无需外网**）。

    Args:
        host / port / timeout / poll_wait / drain_wait: 传给 :class:`MT4BridgeClient`。
            port 缺省从统一设置系统读取（settings.yaml → mt4.port）。
        page_bars: 每次 ``F042`` 请求的根数（1..5000）。
        max_bars: ``fetch_full`` 累计上限，防止无限翻页。
        time_base: 输出时间戳口径，``"utc"``（默认，符合 panel 契约）/
            ``"broker"``（经纪商原始时间）/ ``"shanghai"``（UTC+8，与既有缓存对齐）。
        utc_offset_hours: 经纪商时区偏移（小时）。``None`` 时首次取数用
            :meth:`MT4BridgeClient.detect_utc_offset_hours` 实测并缓存——
            经纪商 DST 切换时不能靠硬编码。
    """

    source_name = "MT4"

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = _settings_mt4_port(),
        timeout: float = 3.0,
        poll_wait: float = 1.6,
        drain_wait: float = 0.25,
        page_bars: int = 500,
        max_bars: int = 20000,
        time_base: str = "utc",
        utc_offset_hours: float | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._timeout = timeout
        self._poll_wait = poll_wait
        self._drain_wait = drain_wait
        self._page_bars = max(1, min(int(page_bars), _EA_MAX_COUNT))
        self._max_bars = max(1, int(max_bars))
        if time_base not in _TIME_BASES:
            raise DataError(
                f"非法 time_base: {time_base!r}", context={"allowed": list(_TIME_BASES)}
            )
        self._time_base = time_base
        self._offset_cfg = utc_offset_hours
        self._offset_cache: float | None = None
        self._avail_cache: bool | None = None

    # ── 基类接口 ─────────────────────────────────────────────────────────

    def _make_client(self) -> MT4BridgeClient:
        """按实例参数构造客户端（集中一处，避免四个调用点各写一遍）。"""
        return MT4BridgeClient(
            self._host,
            self._port,
            self._timeout,
            self._poll_wait,
            self._drain_wait,
        )

    def is_available(self) -> bool:
        """连得上并 ``F000`` 探活成功才算可用；结果按实例缓存。

        注意：这**只证明 EA 在监听**，不代表有行情数据——终端未接入经纪商时
        EA 仍正常响应，但报价为 0、历史可能为空。可用性判定保持轻量，
        真实数据问题在 :meth:`fetch_full` 里如实报错。
        """
        if self._avail_cache is not None:
            return self._avail_cache
        client = self._make_client()
        ok = False
        try:
            client.connect().ping(wait=self._poll_wait)
            ok = True
        except DataError:
            ok = False
        finally:
            client.close()
        self._avail_cache = ok
        return ok

    def describe(self) -> str:
        base = self._time_base
        off = self._offset_cache if self._offset_cache is not None else self._offset_cfg
        off_s = f"{off:+g}h" if off is not None else "待实测"
        return f"MT4 Bridge: {self._host}:{self._port} (time_base={base}, broker偏移={off_s})"

    def fetch_full(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """全量取 ``symbol`` + ``timeframe`` 的历史 K 线。

        从 offset 0（最新 bar）向前翻页，直到 EA 返回空（触及本地历史边界）或达到
        :attr:`_max_bars`。每页返回时间降序，故逐页**反序后前插**拼接。

        Raises:
            DataError: 连接失败、周期不支持、或最终为空。
        """
        self._check_symbol(symbol)
        timeframe = timeframe.upper()
        self._require_tf(timeframe)
        bars: list[dict[str, Any]] = []
        client = self._make_client()
        try:
            client.connect()
            offset = 0
            while len(bars) < self._max_bars:
                got = client.bars(symbol, timeframe, offset, self._page_bars)
                if not got:
                    break
                bars = list(reversed(got)) + bars  # got 是时间降序
                if len(got) < self._page_bars:
                    break  # 已到历史边界（iTime()==0）
                offset += len(got)
        finally:
            client.close()
        # bars 此时为时间升序（最新在末尾）。翻页以整页为单位，最后一页可能越过
        # _max_bars——须截断到**最新**的 _max_bars 根，否则 page_bars=500 时最多多
        # 返回 499 根（实测 page_bars=60/max_bars=121 会返回 180 根）。
        if len(bars) > self._max_bars:
            bars = bars[-self._max_bars :]
        return self._to_frame(bars, symbol, timeframe)

    def fetch_incremental(self, symbol: str, timeframe: str, since_ts: int) -> pd.DataFrame:
        """取 ``since_ts`` 之后的数据。

        ``F042`` 的 ``offset`` 按**根数**计（0 = 最新），但外汇有每日/每周休市，
        所以「墙上小时数 ÷ 周期」不等于根数——H1 实测 bar 数约为墙上小时数的
        0.57。因此**不做任何比例估算**，改为翻页时逐页检查时间是否已越过
        ``since_ts`` 边界，越过即停：既不因低估而漏最新数据，也不因高估而空转。

        ``since_ts`` 由调用方按 :attr:`_time_base` 口径给出（本项目统一 UTC）。
        """
        self._check_symbol(symbol)
        timeframe = timeframe.upper()
        self._require_tf(timeframe)
        since_broker = int(since_ts) - self._to_broker_delta()

        client = self._make_client()
        try:
            client.connect()
            head = client.bars(symbol, timeframe, 0, 1)
            if not head:
                return self._finalize(pd.DataFrame())
            bars: list[dict[str, Any]] = []
            start = 0
            while start < self._max_bars:
                got = client.bars(symbol, timeframe, start, self._page_bars)
                if not got:
                    break  # 触及本机历史边界（iTime()==0）
                bars.extend(got)
                if len(got) < self._page_bars:
                    break  # 已拉空
                # got 是 offset 升序 = 时间降序，故末条是本页最老的一根
                if int(got[-1]["time"]) <= since_broker:
                    break  # 已越过 since 边界
                start += len(got)
        finally:
            client.close()
        return self._to_frame(bars, symbol, timeframe, since_ts=since_ts)

    # ── 内部 ─────────────────────────────────────────────────────────────

    @staticmethod
    def _check_symbol(symbol: str) -> None:
        s = (symbol or "").strip()
        if not s:
            raise DataError("MT4 取数需要非空品种名", context={"symbol": symbol})
        if "#" in s or "$" in s or "!" in s:
            raise DataError(f"品种名含非法协议字符: {symbol!r}", context={"symbol": symbol})

    @staticmethod
    def _require_tf(timeframe: str) -> None:
        if timeframe not in MT4_TIMEFRAME:
            raise DataError(
                f"MT4 不支持的周期: {timeframe}",
                context={"supported": sorted(MT4_TIMEFRAME)},
            )

    def _resolve_offset_hours(self) -> float:
        """取经纪商时区偏移（小时）；未配置则实测并缓存。"""
        if self._offset_cfg is not None:
            return self._offset_cfg
        if self._offset_cache is not None:
            return self._offset_cache
        try:
            with self._make_client() as client:
                self._offset_cache = client.detect_utc_offset_hours()
        except DataError:
            # 测不到就按 UTC 处理（偏移 0），但保留原始 broker 口径不受影响
            self._offset_cache = 0.0
        return self._offset_cache

    def _to_broker_delta(self) -> int:
        """输出时间基准相对经纪商时间的秒数偏移（output = broker + delta）。"""
        if self._time_base == "broker":
            return 0
        return int(round((_TZ_OFFSET_HOURS[self._time_base] - self._resolve_offset_hours()) * 3600))

    def _to_frame(
        self,
        bars: list[dict[str, Any]],
        symbol: str,
        timeframe: str,
        since_ts: int | None = None,
    ) -> pd.DataFrame:
        """把原始 bar 列表转成统一契约 DataFrame（含时间基准换算）。

        Args:
            since_ts: 给定时只保留 ``time > since_ts`` 的行（增量取数用）。
                ⚠ 必须在**换算时间基准之后**过滤——原始经纪商时间无法直接与
                调用方的 ``since_ts``（UTC 口径）比较。
        """
        delta = self._to_broker_delta()
        for b in bars:
            b["time"] = int(b["time"]) + delta
        df = pd.DataFrame(bars)
        if since_ts is not None:
            df = df[df["time"] > int(since_ts)]
        if len(df) == 0:
            if not bars:
                raise DataError(
                    f"MT4 无数据: {symbol} {timeframe}",
                    context={
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "host": self._host,
                        "port": self._port,
                        "hint": "品种名需与终端一致；或该周期历史尚未下载到本机",
                    },
                )
            # 有数据但全被 since 过滤掉 = 确实没有新数据，返回空（非错误）
            return self._finalize(pd.DataFrame())
        df = df.rename(columns={"volume": "tick_volume"})
        df["volume"] = df["tick_volume"]
        return self._finalize(df)
