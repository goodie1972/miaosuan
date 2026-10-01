import sys, time, traceback
out = open(r'D:\backup\BaoBao\PythonProgram\miaosuan\test_out7.log', 'w', encoding='utf-8')
def log(*args):
    msg = ' '.join(str(a) for a in args)
    print(msg, file=out, flush=True)

try:
    log('Python OK')
    sys.path.insert(0, 'src')

    # Test 1: filename pattern computation (fast, no network)
    from miaosuan.data.acquisition import DataAcquisition
    acq = DataAcquisition()
    
    log('=== Filename pattern tests ===')
    p1 = acq._cache_path('XAUUSD', 'D1', note=None, source_type='Dukascopy')
    log(f'source only: {p1.name}')
    
    p2 = acq._cache_path('XAUUSD', 'D1', note='test', source_type=None)
    log(f'note only: {p2.name}')
    
    p3 = acq._cache_path('XAUUSD', 'D1', note='test', source_type='Dukascopy')
    log(f'both: {p3.name}')
    
    p4 = acq._cache_path('XAUUSD', 'H1', note='my test note', source_type='Dukascopy')
    log(f'special chars: {p4.name}')

    # Test 2: D1 via DataAcquisition (already cached from previous test)
    log('\n=== DataAcquisition D1 fetch (cached) ===')
    t0 = time.time()
    path = acq.fetch('XAUUSD', 'D1', source='Dukascopy', note='test', allowed_types=['Dukascopy'])
    log(f'Cached in {time.time()-t0:.1f}s to: {path}')
    log(f'Filename: {path.name}')
    log(f'Contains source: {"Dukascopy" in path.name}')
    log(f'Contains note: {"test" in path.name}')

    # Test 3: Incremental fetch (should use max time from cache)
    log('\n=== Incremental fetch test ===')
    from miaosuan.data.fetchers.dukascopy import DukascopyFetcher, _parse_response
    import pandas as pd
    
    # Read existing cache
    if path.exists():
        old_df = pd.read_parquet(path)
        max_time = int(old_df['time'].max())
        log(f'Cache max_time={max_time}')
        log(f'Cache rows={len(old_df)}')
        log(f'Last row: {old_df.tail(1).to_dict("records")[0]}')

    # Test 4: Direct API test for the latest available data
    log('\n=== Direct API test ===')
    import urllib.request, json
    proxy = 'http://127.0.0.1:10808'
    handler = urllib.request.ProxyHandler({'http': proxy, 'https': proxy})
    opener = urllib.request.build_opener(handler)
    
    # Check hour data for 2025-12 (should have data)
    url = 'https://jetta.dukascopy.com/v1/candles/hour/XAU-USD/BID/2025/12'
    log(f'trying: {url}')
    req = urllib.request.Request(url, headers={'User-Agent': 'MiaoSuan/1.0'})
    with opener.open(req, timeout=30) as resp:
        body = resp.read()
        data = json.loads(body.decode('utf-8'))
        log(f'hour 2025-12: times={len(data.get("times",[]))}')
    
    # Check hour data for 2026-01 (should 400)
    url = 'https://jetta.dukascopy.com/v1/candles/hour/XAU-USD/BID/2026/1'
    log(f'trying: {url}')
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'MiaoSuan/1.0'})
        with opener.open(req, timeout=30) as resp:
            body = resp.read()
            data = json.loads(body.decode('utf-8'))
            log(f'hour 2026-01: times={len(data.get("times",[]))}')
    except Exception as e:
        log(f'hour 2026-01: ERROR {e}')

    # Test 5: M5 with 1-day window (single day, should be fast)
    log('\n=== M5 1-day test ===')
    f = DukascopyFetcher(proxy='http://127.0.0.1:10808', timeout=30, window_days=3)
    # Manually set fetch range to last 3 days
    from datetime import datetime, timezone
    f._fetch_start_dt = datetime(2025, 12, 28, tzinfo=timezone.utc)
    f._fetch_end_dt = datetime(2025, 12, 30, tzinfo=timezone.utc)
    f._fetch_tf = 'M5'
    t0 = time.time()
    raw = f._fetch_timeframe('XAU-USD', 'M5', [2025])
    log(f'M5 2 days: {time.time()-t0:.1f}s, raw_rows={len(raw)}')
    if raw:
        log(f'First: {raw[0]}')
        log(f'Last: {raw[-1]}')

except Exception as e:
    log(f'ERROR: {e}')
    traceback.print_exc(file=out)
finally:
    out.close()
