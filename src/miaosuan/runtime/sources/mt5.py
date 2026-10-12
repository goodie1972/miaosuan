# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for full license text.

"""MT5 直连 DataPort（runtime 数据通道第一实现，复用 R1 的 fetcher）。"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

from ...data.fetchers import MT5Fetcher
from ..ports import Bar, DataPort
from ..scheduler import timeframe_seconds

if TYPE_CHECKING:  # 仅类型提示
    from ...market.profiles import BrokerSpec

__all__ = ["MT5DataPort"]


class MT5DataPort(DataPort):
    """把 :class:`~miaosuan.data.fetchers.MT5Fetcher` 适配成 runtime 的 DataPort。

    收盘判定（端口层纪律）：``bar.ts + tf_seconds <= now_utc`` 才产出——
    ``copy_rates`` 的最后一根是 forming bar，绝不入流。断线补拉由 fetcher
    的增量语义承担（``fetch_incremental(since_ts)`` 不重不漏），本类再按
    ``ts`` 水位去重一道（调度层还有第三道）。

    :param now_fn: 当前 UTC 秒的取法（默认 ``time.time``；测试注入假时钟）。
    """

    def __init__(
        self,
        fetcher: MT5Fetcher | None = None,
        *,
        now_fn: Callable[[], float] = time.time,
    ) -> None:
        self._fetcher = fetcher or MT5Fetcher()
        self._now_fn = now_fn
        self._watermarks: dict[str, int] = {}

    def stream_closed_bars(self, symbol: str, timeframe: str) -> Iterator[Bar]:
        """拉取自上次调用以来新收盘的 bar；首次调用回填终端可用历史。"""
        label = f"{symbol}.{timeframe}"
        since = self._watermarks.get(label, 0)
        tf_sec = timeframe_seconds(timeframe)
        df = (
            self._fetcher.fetch_incremental(symbol, timeframe, since_ts=since)
            if since
            else self._fetcher.fetch_full(symbol, timeframe)
        )
        max_ts = since
        now = int(self._now_fn())
        for row in df.itertuples(index=False):
            ts = int(row.time)
            if ts + tf_sec > now:
                continue  # forming bar（端口层收盘纪律）
            if ts <= max_ts:
                continue  # 水位去重
            yield Bar(
                symbol=symbol,
                timeframe=timeframe,
                ts=ts,
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=float(row.volume),
                closed=True,
            )
            max_ts = ts
        self._watermarks[label] = max_ts

    def symbol_spec(self, symbol: str) -> BrokerSpec:
        """从终端读品种规格（R4 的 ``MT5Broker`` 落地时实现）。"""
        raise NotImplementedError("品种规格读取在 R4（MT5Broker）落地")
