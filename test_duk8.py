import sys, time, traceback
out = open(r'D:\backup\BaoBao\PythonProgram\miaosuan\test_out8.log', 'w', encoding='utf-8')
def log(*args):
    msg = ' '.join(str(a) for a in args)
    print(msg, file=out, flush=True)

try:
    log('Python OK')
    sys.path.insert(0, 'src')

    # Step 1: Check proxy availability
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(3)
    try:
        s.connect(('127.0.0.1', 10808))
        log('Proxy port 10808: REACHABLE')
    except:
        log('Proxy port 10808: NOT REACHABLE')
    finally:
        s.close()

    # Step 2: Check config proxy
    from miaosuan.data.acquisition import DataAcquisition, data_acquisition_config
    cfg = data_acquisition_config()
    log(f'cfg.proxy = {cfg.proxy}')
    log(f'cfg.timeout = {cfg.timeout}')
    log(f'cfg.dukascopy_user = {cfg.dukascopy_user}')
    log(f'cfg.dukascopy_password = {cfg.dukascopy_password}')

    # Step 3: Create DataAcquisition and inspect its sources
    acq = DataAcquisition()
    sources = acq._select_sources('Dukascopy', allowed_types=['Dukascopy'])
    log(f'\nSelected sources: {[type(s).__name__ for s in sources]}')
    
    for src in sources:
        f = getattr(src, '_fetcher', None)
        if f is not None:
            log(f'  inner fetcher: {type(f).__name__}')
            log(f'  proxy: {getattr(f, "_proxy", "N/A")}')
            log(f'  timeout: {getattr(f, "_timeout", "N/A")}')
            log(f'  window_days: {getattr(f, "_window_days", "N/A")}')
        log(f'  describe: {src.describe()}')
        log(f'  is_available: {src.is_available()}')

    # Step 4: Manually call fetch_full and catch the actual exception
    log('\n=== Manual fetch_full test ===')
    for src in sources:
        try:
            log(f'Calling fetch_full on {type(src).__name__}...')
            t0 = time.time()
            df = src.fetch_full('XAUUSD', 'D1')
            log(f'  SUCCESS in {time.time()-t0:.1f}s, rows={len(df)}')
        except Exception as e:
            log(f'  FAILED in {time.time()-t0:.1f}s: {type(e).__name__}: {e}')

    # Step 5: Now test _try_fetch_full with verbose logging
    log('\n=== _try_fetch_full test ===')
    t0 = time.time()
    result = acq._try_fetch_full(sources, 'XAUUSD', 'D1')
    if result is not None:
        log(f'  SUCCESS: rows={len(result)} in {time.time()-t0:.1f}s')
    else:
        log(f'  FAILED (returned None) in {time.time()-t0:.1f}s')

    # Step 6: Try fetch_incremental
    log('\n=== _try_fetch_incremental test ===')
    t0 = time.time()
    result = acq._try_fetch_incremental(sources, 'XAUUSD', 'D1', 1700000000)
    if result is not None:
        log(f'  SUCCESS: rows={len(result)} in {time.time()-t0:.1f}s')
    else:
        log(f'  FAILED (returned None) in {time.time()-t0:.1f}s')

except Exception as e:
    log(f'ERROR: {e}')
    traceback.print_exc(file=out)
finally:
    out.close()
