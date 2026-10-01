import sys, time, traceback
out = open(r'D:\backup\BaoBao\PythonProgram\miaosuan\test_out9.log', 'w', encoding='utf-8')
def log(*args):
    msg = ' '.join(str(a) for a in args)
    print(msg, file=out, flush=True)

try:
    log('Python OK')
    sys.path.insert(0, 'src')
    from miaosuan.data.acquisition import DataAcquisition
    from miaosuan.errors import DataError

    acq = DataAcquisition()
    sources = acq._select_sources('Dukascopy', allowed_types=['Dukascopy'])
    src = sources[0]
    f = src._fetcher

    # Directly call DukascopyFetcher.fetch_full to see the raw error
    log('=== Direct DukascopyFetcher.fetch_full ===')
    t0 = time.time()
    try:
        df = f.fetch_full('XAUUSD', 'D1')
        log(f'SUCCESS in {time.time()-t0:.1f}s, rows={len(df)}')
    except DataError as e:
        log(f'DataError in {time.time()-t0:.1f}s:')
        log(f'  code={e.code}')
        log(f'  message={e.message}')
        log(f'  context={e.context}')
        log(f'  str(e)={str(e)}')
    except Exception as e:
        log(f'Other error in {time.time()-t0:.1f}s: {type(e).__name__}: {str(e)}')

    # Also check _fetch_raw directly
    log('\n=== Direct _fetch_raw ===')
    t0 = time.time()
    try:
        raw = f._fetch_raw('XAUUSD', 86400)
        log(f'_fetch_raw returned {len(raw)} rows in {time.time()-t0:.1f}s')
        if raw:
            log(f'First: {raw[0]}')
            log(f'Last: {raw[-1]}')
    except Exception as e:
        log(f'_fetch_raw error in {time.time()-t0:.1f}s: {type(e).__name__}: {str(e)}')

except Exception as e:
    log(f'ERROR: {e}')
    traceback.print_exc(file=out)
finally:
    out.close()
