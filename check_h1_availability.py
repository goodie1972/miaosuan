import sys, time, json, urllib.request
from datetime import datetime, timezone

proxy = 'http://127.0.0.1:10808'
handler = urllib.request.ProxyHandler({'http': proxy, 'https': proxy})
opener = urllib.request.build_opener(handler)

print("=== 检查 Dukascopy H1 可用月份 ===")
for year in [2024, 2025, 2026]:
    for month in range(1, 13):
        if year == 2026 and month > 9:  # 当前只到 9 月
            continue
        url = f'https://jetta.dukascopy.com/v1/candles/hour/XAU-USD/BID/{year}/{month}'
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'MiaoSuan/1.0'})
            with opener.open(req, timeout=15) as resp:
                body = resp.read()
                data = json.loads(body.decode('utf-8'))
                n = len(data.get('times', []))
                if n > 0:
                    print(f"  {year}-{month:02d}: {n} 根 OK")
                else:
                    print(f"  {year}-{month:02d}: 空")
        except Exception as e:
            if '400' in str(e):
                print(f"  {year}-{month:02d}: 400 (未就绪)")
            else:
                print(f"  {year}-{month:02d}: ERROR {e}")

print("\n=== 检查 M5 近期几天 ===")
import calendar
for day in range(25, 31):
    url = f'https://jetta.dukascopy.com/v1/candles/minute/XAU-USD/BID/2026/9/{day}'
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'MiaoSuan/1.0'})
        with opener.open(req, timeout=15) as resp:
            body = resp.read()
            data = json.loads(body.decode('utf-8'))
            n = len(data.get('times', []))
            dt = datetime(2026, 9, day, tzinfo=timezone.utc)
            if dt.weekday() < 5:
                print(f"  2026-09-{day:02d} ({['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][dt.weekday()]}): {n} 根")
    except Exception as e:
        if '400' in str(e):
            print(f"  2026-09-{day:02d}: 400")
        else:
            print(f"  2026-09-{day:02d}: ERROR {e}")