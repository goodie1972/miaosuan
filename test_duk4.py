import sys, time, json, urllib.request
out = open(r'D:\backup\BaoBao\PythonProgram\miaosuan\test_out4.log', 'w', encoding='utf-8')
def log(*args):
    msg = ' '.join(str(a) for a in args)
    print(msg, file=out, flush=True)

proxy = 'http://127.0.0.1:10808'
handler = urllib.request.ProxyHandler({'http': proxy, 'https': proxy})
opener = urllib.request.build_opener(handler)

# Test API for year 2026
for year in [2026, 2025]:
    url = f'https://jetta.dukascopy.com/v1/candles/day/XAU-USD/BID/{year}'
    log(f'\n--- Year {year} ---')
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'MiaoSuan/1.0'})
        with opener.open(req, timeout=30) as resp:
            body = resp.read()
            data = json.loads(body.decode('utf-8'))
            ts_len = len(data.get('times', []))
            log(f'Status: 200, times={ts_len}')
            if ts_len > 0:
                # Convert timestamp
                from datetime import datetime, timezone
                base_ts = data.get('timestamp', 0)
                shift = data.get('shift', 0)
                # First bar time
                first_bar = datetime.fromtimestamp(base_ts / 1000, tz=timezone.utc)
                last_idx = ts_len - 1
                last_bar = datetime.fromtimestamp((base_ts + sum(data['times'][:last_idx+1]) * shift) / 1000, tz=timezone.utc)
                log(f'First bar: {first_bar.isoformat()}')
                log(f'Last bar: {last_bar.isoformat()}')
                log(f'Multiplier: {data.get("multiplier")}')
                log(f'Shift: {shift}')
    except Exception as e:
        log(f'ERROR: {e}')

# Now test with _parse_response
sys.path.insert(0, 'src')
from miaosuan.data.fetchers.dukascopy import DukascopyFetcher, _parse_response
from datetime import datetime, timezone

url = 'https://jetta.dukascopy.com/v1/candles/day/XAU-USD/BID/2026'
req = urllib.request.Request(url, headers={'User-Agent': 'MiaoSuan/1.0'})
with opener.open(req, timeout=30) as resp:
    body = resp.read()
    data = json.loads(body.decode('utf-8'))
    rows = _parse_response(data, 86400)
    log(f'\n--- _parse_response for 2026 ---')
    log(f'rows={len(rows)}')
    if rows:
        log(f'First: {rows[0]}')
        log(f'Last: {rows[-1]}')
        # Filter like _fetch_timeframe does
        start_dt = datetime(2026, 9, 15, tzinfo=timezone.utc)
        end_dt = datetime(2026, 9, 29, tzinfo=timezone.utc)
        start_ts = int(start_dt.timestamp())
        end_ts = int(end_dt.timestamp())
        log(f'Filter: start_ts={start_ts}, end_ts={end_ts}')
        filtered = [r for r in rows if r["time"] >= start_ts and r["time"] <= end_ts]
        log(f'Filtered rows={len(filtered)}')

out.close()
