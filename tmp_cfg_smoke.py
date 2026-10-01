# -*- coding: utf-8 -*-
"""配置保存冒烟：复现事故 payload（8 个数值字段全 null）→ 必须 200（一次性脚本）。"""
import json
import urllib.error
import urllib.request

PAYLOAD = {
    "webui": {"host": "127.0.0.1", "port": None, "reload": False},
    "shenji": {"host": "127.0.0.1", "port": None, "timeout": None},
    "mt4": {
        "host": "127.0.0.1",
        "port": None,
        "time_base": "utc",
        "timeout": None,
        "poll_wait": None,
        "drain_wait": None,
    },
    "data": {
        "timeout": None,
        "source": "",
        "api_url": "",
        "cache_dir": "",
        "dukascopy_user": "",
        "dukascopy_password": "",
    },
}


def post(payload):
    req = urllib.request.Request(
        "http://127.0.0.1:8686/api/config",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        r = urllib.request.urlopen(req, timeout=30)
        return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:300]


out = []
# 1) 事故 payload（全 null）→ 修复后必须 200
code, body = post(PAYLOAD)
out.append(f"[1] all-null payload -> {code}")
if code != 200:
    out.append(f"    FAIL detail: {body}")

# 2) 正常更新一个字段 → 200 且生效
code, body = post({"webui": {"port": 8686}})
out.append(f"[2] normal update -> {code}")
if code != 200:
    out.append(f"    FAIL detail: {body}")

# 3) GET 回读确认配置完好
req = urllib.request.Request("http://127.0.0.1:8686/api/config")
cfg = json.loads(urllib.request.urlopen(req, timeout=30).read().decode())
out.append(f"[3] GET webui.port={cfg['webui']['port']} shenji.port={cfg['shenji']['port']} "
           f"mt4.port={cfg['mt4']['port']} data.timeout={cfg['data']['timeout']}")
ok = cfg["webui"]["port"] == 8686 and cfg["shenji"]["port"] == 1783
out.append(f"[3] intact={'YES' if ok else 'NO'}")

with open("cfg_smoke_out.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print("done")
