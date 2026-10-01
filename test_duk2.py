import sys, time, traceback
out = open(r'D:\backup\BaoBao\PythonProgram\miaosuan\test_out2.log', 'w', encoding='utf-8')
def log(*args):
    msg = ' '.join(str(a) for a in args)
    print(msg, file=out, flush=True)

try:
    log('Python OK')
    sys.path.insert(0, 'src')
    from miaosuan.data.fetchers.dukascopy import DukascopyFetcher
    log('import OK')

    f = DukascopyFetcher(proxy='http://127.0.0.1:10808', window_days=14, timeout=30)
    log(f'fetcher created, window_days={f._window_days}')
    log(f'is_available={f.is_available()}')

    # Try D1 with small window (fastest: 1 request)
    log('=== D1 test ===')
    t0 = time.time()
    df = f.fetch_full('XAUUSD', 'D1')
    log(f'D1 done in {time.time()-t0:.1f}s, rows={len(df)}')
    if len(df) > 0:
        log(df.head(2).to_string())
        log(df.tail(2).to_string())

    # Try M5 with small window
    log('=== M5 test ===')
    t0 = time.time()
    df = f.fetch_full('XAUUSD', 'M5')
    log(f'M5 done in {time.time()-t0:.1f}s, rows={len(df)}')
    if len(df) > 0:
        log(df.head(2).to_string())

    # Test via DataAcquisition with note + source
    log('=== DataAcquisition test ===')
    from miaosuan.data.acquisition import DataAcquisition
    acq = DataAcquisition()
    path = acq.fetch('XAUUSD', 'D1', source='Dukascopy', note='test', allowed_types=['Dukascopy'])
    log(f'Success! Cached to: {path}')
    log(f'Filename: {path.name}')

except Exception as e:
    log(f'ERROR: {e}')
    traceback.print_exc(file=out)
finally:
    out.close()
