"""
US Stock Hourly Scraper — parallel batch download, 730-day max (yfinance limit)
Saves to core/datasets/us_stocks_1h.parquet
"""

import warnings
warnings.filterwarnings('ignore')

import time
import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── EDIT ─────────────────────────────────────
OUTPUT   = Path('core/datasets/us_stocks_1h.parquet')
WORKERS  = 3      # parallel batch threads
BATCH    = 50     # tickers per batch
SLEEP    = 2      # seconds between batches
DAYS     = 729
# ─────────────────────────────────────────────

def get_tickers() -> list[str]:
    try:
        nasdaq = pd.read_csv('https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt', sep='|')
        other  = pd.read_csv('https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt',  sep='|')
        tickers = (
            nasdaq[nasdaq['Financial Status'] == 'N']['Symbol'].tolist() +
            other[other['ETF'] == 'N']['ACT Symbol'].tolist()
        )
        tickers = [t for t in tickers if isinstance(t, str) and len(t) <= 5 and t.isalpha()]
        print(f"  {len(tickers)} tickers from Nasdaq listing")
        return tickers
    except Exception as e:
        print(f"  Nasdaq failed ({e}), using S&P 500")
        table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        return table['Symbol'].str.replace('.', '-', regex=False).tolist()


def fetch_batch(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    for attempt in range(3):
        try:
            raw = yf.download(
                tickers, start=start, end=end,
                interval='1h', group_by='ticker',
                auto_adjust=True, progress=False, threads=True,
            )
            break
        except Exception:
            time.sleep(5 * (attempt + 1))
    else:
        return pd.DataFrame()

    if raw.empty:
        return pd.DataFrame()

    raw.columns = raw.columns.swaplevel(0, 1)
    raw.sort_index(axis=1, inplace=True)

    frames = []
    for t in tickers:
        try:
            df = raw[t].dropna(how='all').reset_index()
            df.insert(0, 'ticker', t)
            frames.append(df)
        except Exception:
            pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main():
    end   = datetime.now()
    start = end - timedelta(days=DAYS)
    start_str = start.strftime('%Y-%m-%d')
    end_str   = end.strftime('%Y-%m-%d')

    print(f"\nFetching {start_str} → {end_str}")
    tickers = get_tickers()
    batches = [tickers[i:i+BATCH] for i in range(0, len(tickers), BATCH)]
    print(f"  {len(batches)} batches × {BATCH} tickers | {WORKERS} workers\n")

    all_frames = []
    done = 0

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {}
        for i, b in enumerate(batches):
            futures[pool.submit(fetch_batch, b, start_str, end_str)] = b
            if i % WORKERS == 0:
                time.sleep(SLEEP)
        for future in as_completed(futures):
            df = future.result()
            if not df.empty:
                all_frames.append(df)
            done += 1
            if done % 10 == 0 or done == len(batches):
                rows = sum(len(f) for f in all_frames)
                print(f"  [{done}/{len(batches)}] {rows:,} rows collected", end='\r')

    print()
    if not all_frames:
        print("No data collected.")
        return

    combined = pd.concat(all_frames, ignore_index=True)
    combined.sort_values(['ticker', 'Datetime'], inplace=True)
    combined.drop_duplicates(subset=['ticker', 'Datetime'], inplace=True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUTPUT, index=False, compression='snappy')

    size_mb = OUTPUT.stat().st_size / 1e6
    print(f"\nSaved {len(combined):,} rows — {size_mb:.1f} MB → {OUTPUT}")
    print(f"Tickers with data: {combined['ticker'].nunique():,}")


if __name__ == '__main__':
    main()
