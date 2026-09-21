# -*- coding: utf-8 -*-
# Copyright (c) 2026 MiaoSuan Team
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file in the project root for full license text.

"""Dukascopy 公开历史接口连通性自检脚本。

用途：在你**自己的机器**上确认 MiaoSuan 的 Dukascopy 数据源能否取到真实行情。

用法（在项目根目录，用项目虚拟环境）::

    .venv\\Scripts\\python.exe scripts/check_dukascopy.py
    .venv\\Scripts\\python.exe scripts/check_dukascopy.py XAUUSD H1

要点：
* Dukascopy 的**历史**蜡烛接口是**公开免登录**的，本脚本不发任何账号密码；
  所以模拟 / 实盘账户是否过期，与本脚本的结论**无关**。
* 若 ``主机可达 = False``，属于网络层不通（公司网络 / 代理 / 防火墙 /
  运营商 DNS 都可能），换网络或配代理后再跑一次即可。
"""

from __future__ import annotations

import sys
import time

# 允许以脚本方式直接运行（不必先 pip install -e）
if __name__ == "__main__" and __package__ is None:  # pragma: no cover
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from miaosuan.data.fetchers.dukascopy import DukascopyFetcher  # noqa: E402
from miaosuan.errors import DataError  # noqa: E402


def main(argv: list[str]) -> int:
    symbol = argv[1] if len(argv) > 1 else "XAUUSD"
    timeframe = argv[2] if len(argv) > 2 else "D1"

    fetcher = DukascopyFetcher()

    print("=" * 60)
    print("Dukascopy 公开历史接口自检")
    print("=" * 60)
    print(f"标的     : {symbol}")
    print(f"周期     : {timeframe}")
    print("接口     : https://freeserv.dukascopy.com/2.0/rest/ps/public/candles")
    print("凭据     : 无（公开接口，模拟账户过期不影响取数）")
    print("-" * 60)

    t0 = time.time()
    reachable = fetcher.is_available()
    print(f"主机可达 : {reachable}  (探测耗时 {time.time() - t0:.2f}s)")
    print(f"来源描述 : {fetcher.describe()}")
    print("-" * 60)

    if not reachable:
        print("结论：网络层不通，取数必然失败。")
        print("建议：换网络 / 关闭或正确配置代理 / 检查 DNS 后重跑本脚本。")
        return 2

    t0 = time.time()
    try:
        df = fetcher.fetch_full(symbol, timeframe)
    except DataError as exc:
        print(f"取数失败 : {exc}")
        print("-" * 60)
        print("结论：主机通了但没取到数据，请检查标的/周期是否被支持。")
        return 3

    elapsed = time.time() - t0
    print(f"取数成功 : {len(df)} 根  (耗时 {elapsed:.2f}s)")
    print("-" * 60)
    print(f"列       : {list(df.columns)}")
    if len(df) > 0:
        try:
            print(df.head(3).to_string())
            print("...")
            print(df.tail(3).to_string())
        except Exception:  # pragma: no cover - 仅展示用
            print(df.head(3))
    print("-" * 60)
    print("结论：Dukascopy 数据源在你本机可用，可在 WebUI「数据获取」处直接使用。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
