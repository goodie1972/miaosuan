# -*- coding: utf-8 -*-
"""追加模式成功路径冒烟（一次性脚本，跑完即删）。"""
import json
import urllib.error
import urllib.request

req = urllib.request.Request(
    "http://127.0.0.1:8686/api/acquisition/fetch",
    data=json.dumps(
        {
            "symbol": "XAUUSD",
            "timeframe": "D1",
            "source": "Dukascopy",
            "append_to": r"D:\K线数据\XAUUSD_D1.parquet",
        }
    ).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
try:
    r = urllib.request.urlopen(req, timeout=120)
    d = json.loads(r.read().decode())
    out = [
        f'status={d.get("status")}',
        f'path={d.get("path")}',
        f'source_used={d.get("source_used")}',
        f'errors={d.get("errors")}',
        f'warnings={d.get("warnings")}',
        f'append_to={d.get("append_to")}',
    ]
except urllib.error.HTTPError as e:
    out = [f"HTTP {e.code}", e.read().decode()[:500]]

with open("smoke_ap3.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print("done")
