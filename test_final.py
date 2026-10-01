import sys, time, logging
logging.basicConfig(level=logging.WARNING)
sys.path.insert(0, 'src')

# Test 1: Filename pattern (no network)
from miaosuan.data.acquisition import DataAcquisition
acq = DataAcquisition()

p = acq._cache_path('XAUUSD', 'D1', note='test', source_type='Dukascopy')
print(f'Filename pattern: {p.name}')
assert p.name == 'XAUUSD_D1_Dukascopy_test.parquet', f'Unexpected: {p.name}'

p = acq._cache_path('XAUUSD', 'D1', note='my note with spaces', source_type='Dukascopy')
print(f'Special chars: {p.name}')
assert p.name == 'XAUUSD_D1_Dukascopy_my_note_with_spaces.parquet'

# Test 2: Direct API test (proxy available)
import urllib.request, json
proxy = 'http://127.0.0.1:10808'
handler = urllib.request.ProxyHandler({'http': proxy, 'https': proxy})
opener = urllib.request.build_opener(handler)

url = 'https://jetta.dukascopy.com/v1/candles/day/XAU-USD/BID/2025'
req = urllib.request.Request(url, headers={'User-Agent': 'MiaoSuan/1.0'})
with opener.open(req, timeout=30) as resp:
    body = resp.read()
    data = json.loads(body.decode('utf-8'))
    print(f'API 2025: times={len(data["times"])}')

# Test 3: Fetch via DataAcquisition with logging enabled
from miaosuan.data.fetchers.dukascopy import DukascopyFetcher
f = DukascopyFetcher(proxy=proxy, timeout=30)

# Remove old cache to force fresh fetch
import os
cache_path = acq._cache_path('XAUUSD', 'H1', note='quick_test', source_type='Dukascopy')
if cache_path.exists():
    os.remove(cache_path)
    print(f'Removed old cache: {cache_path}')

print(f'\nFetching H1 with source+note...')
t0 = time.time()
path = acq.fetch('XAUUSD', 'H1', source='Dukascopy', note='quick_test', allowed_types=['Dukascopy'])
elapsed = time.time() - t0
print(f'Done in {elapsed:.1f}s -> {path}')
print(f'Filename: {path.name}')
assert 'Dukascopy' in path.name, 'Source missing from filename'
assert 'quick_test' in path.name, 'Note missing from filename'

# Read back
import pandas as pd
df = pd.read_parquet(path)
print(f'Rows: {len(df)}')
if len(df) > 0:
    print(df.head(2).to_string())
    print(df.tail(2).to_string())

print('\n=== ALL TESTS PASSED ===')
