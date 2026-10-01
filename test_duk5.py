import sys, time, traceback
out = open(r'D:\backup\BaoBao\PythonProgram\miaosuan\test_out5.log', 'w', encoding='utf-8')
def log(*args):
    msg = ' '.join(str(a) for a in args)
    print(msg, file=out, flush=True)

try:
    log('Python OK')
    sys.path.insert(0, 'src')
    from miaosuan.data.fetchers.dukascopy import DukascopyFetcher
    log('import OK')

    # Use default window (7300 days = 20 years) for D1
    f = DukascopyFetcher(proxy='http://127.0.0.1:10808', timeout=30)
    log(f'window_days={f._window_days}')

    # D1: only 1 request per year → fast
    log('=== D1 test (20-year window) ===')
    t0 = time.time()
    df = f.fetch_full('XAUUSD', 'D1')
    log(f'D1 done in {time.time()-t0:.1f}s, rows={len(df)}')
    if len(df) > 0:
        log(df.head(2).to_string())
        log(df.tail(2).to_string())

    # Now test via DataAcquisition with default window (uses full 20-year)
    log('\n=== DataAcquisition D1 test ===')
    from miaosuan.data.acquisition import DataAcquisition
    acq = DataAcquisition()
    t0 = time.time()
    path = acq.fetch('XAUUSD', 'D1', source='Dukascopy', note='test', allowed_types=['Dukascopy'])
    log(f'Cached in {time.time()-t0:.1f}s to: {path}')
    log(f'Filename: {path.name}')

    # Test the note + source combination
    log('\n=== DataAcquisition with note + source ===')
    t0 = time.time()
    path2 = acq.fetch('XAUUSD', 'H4', source='Dukascopy', note='test', allowed_types=['Dukascopy'])
    log(f'Cached in {time.time()-t0:.1f}s to: {path2}')
    log(f'Filename: {path2.name}')

    # Verify note appears in filename
    log(f'\nFilename contains note: {"test" in path2.name}')
    log(f'Filename contains source: {"Dukascopy" in path2.name}')

except Exception as e:
    log(f'ERROR: {e}')
    traceback.print_exc(file=out)
finally:
    out.close()
