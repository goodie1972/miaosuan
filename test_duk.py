import sys, traceback
out = open(r'D:\backup\BaoBao\PythonProgram\miaosuan\test_out.log', 'w', encoding='utf-8')
def log(*args):
    print(*args, file=out, flush=True)
try:
    log('Python OK')
    sys.path.insert(0, 'src')
    from miaosuan.data.acquisition import DataAcquisition
    log('import OK')
    acq = DataAcquisition()
    log('acq created')
    sources = acq._select_sources('Dukascopy', allowed_types=['Dukascopy'])
    log(f'sources: {[type(s).__name__ for s in sources]}')
    src = sources[0]
    log(f'fetcher: {type(src._fetcher).__name__}')
    log(f'fetcher timeout: {src._fetcher._timeout}')
    log(f'fetcher proxy: {src._fetcher._proxy}')
    df = src.fetch_full('XAUUSD', 'M5')
    log(f'fetch_full done, rows={len(df)}')
    log(df.head(3).to_string())
except Exception as e:
    log(f'ERROR: {e}')
    traceback.print_exc(file=out)
finally:
    out.close()
