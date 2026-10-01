import sys
print('Python works', flush=True)
sys.path.insert(0, 'src')
print('Path inserted', flush=True)
try:
    from miaosuan.data.acquisition import DataAcquisition
    print('Import successful', flush=True)
    acq = DataAcquisition()
    print('Instance created', flush=True)
    print('=== Fetching XAUUSD M5 with Dukascopy source and note (7 days) ===', flush=True)
    path = acq.fetch('XAUUSD', 'M5', source='Dukascopy', note='test', allowed_types=['Dukascopy'])
    print(f'Success! Cached to: {path}', flush=True)
    print(f'Filename: {path.name}', flush=True)
except Exception as e:
    print(f'Error: {e}', flush=True)
    import traceback
    traceback.print_exc()