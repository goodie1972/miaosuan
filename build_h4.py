import sys
sys.path.insert(0, 'src')
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

CACHE_DIR = Path(r'D:\backup\BaoBao\PythonProgram\miaosuan\data\cache')
SYMBOL = 'XAUUSD'

def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    df = df.copy()
    df['dt'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df = df.set_index('dt').sort_index()
    agg = df.resample(rule).agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
        'volume': 'sum', 'tick_volume': 'sum',
    })
    agg = agg.dropna(subset=['open'])
    agg = agg.reset_index()
    agg['time'] = (agg['dt'].astype('int64') // 1_000_000_000).astype('int64')
    agg = agg.drop(columns=['dt'])
    cols = ['time', 'open', 'high', 'low', 'close', 'volume', 'tick_volume']
    return agg[cols].reset_index(drop=True)

# H4 from H1
h1 = pd.read_parquet(CACHE_DIR / f'{SYMBOL}_H1_Dukascopy_full.parquet')
h4 = resample_ohlcv(h1, '4h')
OUT_H4 = CACHE_DIR / f'{SYMBOL}_H4_ultimate.parquet'
tmp = OUT_H4.with_suffix('.tmp')
h4.to_parquet(tmp, index=False)
tmp.replace(OUT_H4)

print(f'H4: {len(h4)} rows')
print(f'Range: {datetime.fromtimestamp(int(h4.time.min()), tz=timezone.utc)} to {datetime.fromtimestamp(int(h4.time.max()), tz=timezone.utc)}')
print(f'File: {OUT_H4}')