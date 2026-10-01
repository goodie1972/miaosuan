import sys, time, traceback
out = open(r'D:\backup\BaoBao\PythonProgram\miaosuan\test_out6.log', 'w', encoding='utf-8')
def log(*args):
    msg = ' '.join(str(a) for a in args)
    print(msg, file=out, flush=True)

try:
    log('Python OK')
    sys.path.insert(0, 'src')

    # Test 1: filename pattern computation (fast, no network)
    from miaosuan.data.acquisition import DataAcquisition, _slugify
    log('=== Filename pattern tests ===')
    acq = DataAcquisition()
    
    # With source only
    p1 = acq._cache_path('XAUUSD', 'D1', note=None, source_type='Dukascopy')
    log(f'source only: {p1.name}')
    
    # With note only
    p2 = acq._cache_path('XAUUSD', 'D1', note='test', source_type=None)
    log(f'note only: {p2.name}')
    
    # With both source and note
    p3 = acq._cache_path('XAUUSD', 'D1', note='test', source_type='Dukascopy')
    log(f'both: {p3.name}')
    
    # With note containing spaces/special chars
    p4 = acq._cache_path('XAUUSD', 'H1', note='my test note', source_type='Dukascopy')
    log(f'special chars: {p4.name}')

    # Test 2: Quick H1 fetch with source + note (1 hour window)
    log('\n=== H1 quick fetch test ===')
    from miaosuan.data.fetchers.dukascopy import DukascopyFetcher
    f = DukascopyFetcher(proxy='http://127.0.0.1:10808', timeout=30, window_days=2)
    t0 = time.time()
    df = f.fetch_full('XAUUSD', 'H1')
    log(f'H1 done in {time.time()-t0:.1f}s, rows={len(df)}')
    if len(df) > 0:
        log(df.head(2).to_string())
        log(df.tail(2).to_string())

    # Test 3: DataAcquisition fetch with source + note (small window)
    log('\n=== DataAcquisition H1 fetch with source+note ===')
    # Remove any old cache for this test
    import os
    cache_path = acq._cache_path('XAUUSD', 'H1', note='test_note', source_type='Dukascopy')
    if cache_path.exists():
        os.remove(cache_path)
        log(f'Removed old cache: {cache_path}')

    t0 = time.time()
    path = acq.fetch('XAUUSD', 'H1', source='Dukascopy', note='test_note', allowed_types=['Dukascopy'])
    log(f'Cached in {time.time()-t0:.1f}s to: {path}')
    log(f'Filename: {path.name}')
    log(f'Filename contains source: {"Dukascopy" in path.name}')
    log(f'Filename contains note: {"test_note" in path.name}')

    # Test 4: Read back the cached data
    log('\n=== Read back cached data ===')
    import pandas as pd
    df2 = pd.read_parquet(path)
    log(f'rows={len(df2)}')
    log(df2.head(2).to_string())
    log(df2.tail(2).to_string())

except Exception as e:
    log(f'ERROR: {e}')
    traceback.print_exc(file=out)
finally:
    out.close()
