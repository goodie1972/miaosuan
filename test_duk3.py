import sys, time, json, urllib.request, urllib.parse
out = open(r'D:\backup\BaoBao\PythonProgram\miaosuan\test_out3.log', 'w', encoding='utf-8')
def log(*args):
    msg = ' '.join(str(a) for a in args)
    print(msg, file=out, flush=True)

# Direct test of Dukascopy API through proxy
try:
    proxy = 'http://127.0.0.1:10808'
    handler = urllib.request.ProxyHandler({'http': proxy, 'https': proxy})
    opener = urllib.request.build_opener(handler)
    log(f'opener built with proxy={proxy}')

    # Test 1: basic connectivity
    url = 'https://jetta.dukascopy.com/v1/candles/day/XAU-USD/BID/2025'
    log(f'trying: {url}')
    t0 = time.time()
    req = urllib.request.Request(url, headers={'User-Agent': 'MiaoSuan/1.0'})
    try:
        with opener.open(req, timeout=30) as resp:
            body = resp.read()
            log(f'GET 200 in {time.time()-t0:.2f}s, {len(body)} bytes')
            data = json.loads(body.decode('utf-8'))
            log(f'keys: {list(data.keys())[:10]}')
            log(f'timestamp: {data.get("timestamp")}, shift: {data.get("shift")}')
            log(f'times len: {len(data.get("times", []))}')
            log(f'volumes len: {len(data.get("volumes", []))}')
            if 'opens' in data:
                log(f'opens len: {len(data["opens"])}')
    except Exception as e:
        log(f'ERROR: {e}')
except Exception as e:
    log(f'OUTER ERROR: {e}')
finally:
    out.close()
