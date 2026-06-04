"""
Generates 1,000,000 synthetic house worker recruitment records.
Target: days_to_arrival (how long from order to worker arriving)
Run once: uv run generate.py
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N   = 1_000_000
OUT = 'workers.csv'

AGENCIES = ['Al Majd', 'Al Rawabi', 'Barq', 'Sama', 'Nakheel',
            'Al Wafa', 'Tadbeer', 'Musaned', 'Al Amal', 'Rayhan']

# base days to arrival per country (realistic Gulf recruitment timelines)
COUNTRY_BASE = {
    'Philippines': 28,
    'Indonesia':   38,
    'Sri Lanka':   45,
    'Ethiopia':    58,
    'India':       22,
    'Bangladesh':  30,
}

HOUSEMAID_COUNTRIES = ['Ethiopia', 'Indonesia', 'Philippines', 'Sri Lanka']
DRIVER_COUNTRIES    = ['India', 'Bangladesh']

# agency additive offset per country — each agency has its own network strength
# positive = slower, negative = faster than country baseline
AGENCY_OFFSETS = {
    'Al Majd':   {'Philippines': -8,  'Indonesia': -5,  'Sri Lanka': -3,  'Ethiopia': -10, 'India': -4,  'Bangladesh': -5},
    'Al Rawabi': {'Philippines': -4,  'Indonesia': -2,  'Sri Lanka':  2,  'Ethiopia':  -5, 'India': -2,  'Bangladesh': -3},
    'Barq':      {'Philippines':  0,  'Indonesia':  3,  'Sri Lanka':  5,  'Ethiopia':   0, 'India':  0,  'Bangladesh':  2},
    'Sama':      {'Philippines':  3,  'Indonesia':  5,  'Sri Lanka':  8,  'Ethiopia':   5, 'India':  3,  'Bangladesh':  4},
    'Nakheel':   {'Philippines':  5,  'Indonesia':  8,  'Sri Lanka': 10,  'Ethiopia':  10, 'India':  5,  'Bangladesh':  6},
    'Al Wafa':   {'Philippines':  8,  'Indonesia': 10,  'Sri Lanka': 12,  'Ethiopia':  15, 'India':  6,  'Bangladesh':  8},
    'Tadbeer':   {'Philippines': -6,  'Indonesia': -4,  'Sri Lanka': -2,  'Ethiopia':  -8, 'India': -3,  'Bangladesh': -4},
    'Musaned':   {'Philippines': -2,  'Indonesia':  0,  'Sri Lanka':  4,  'Ethiopia':   2, 'India': -1,  'Bangladesh':  0},
    'Al Amal':   {'Philippines':  6,  'Indonesia': 12,  'Sri Lanka': 15,  'Ethiopia':  18, 'India':  8,  'Bangladesh': 10},
    'Rayhan':    {'Philippines': 10,  'Indonesia': 15,  'Sri Lanka': 18,  'Ethiopia':  22, 'India': 10,  'Bangladesh': 14},
}

print(f"Generating {N:,} records...")

job     = np.random.choice(['housemaid', 'driver'], N, p=[0.70, 0.30])
country = np.where(
    job == 'housemaid',
    np.random.choice(HOUSEMAID_COUNTRIES, N),
    np.random.choice(DRIVER_COUNTRIES,    N)
)
gender  = np.where(job == 'housemaid', 'F', 'M')
agency  = np.random.choice(AGENCIES, N)

# order dates: 2019-2024
order_date = pd.to_datetime('2019-01-01') + pd.to_timedelta(
    np.random.randint(0, 365 * 5, N), unit='D'
)
order_month      = order_date.month.values
order_day_of_week= order_date.day_of_week.values
order_hour       = np.random.choice(range(8, 22), N)   # business hours

# base days per record
base = np.array([COUNTRY_BASE[c] for c in country], dtype=float)

# agency effect: per-country offset
base += np.array([AGENCY_OFFSETS[a][c] for a, c in zip(agency, country)], dtype=float)

# seasonal effects
ramadan_months = [3, 4]        # approx
summer_months  = [6, 7, 8]
base += np.where(np.isin(order_month, ramadan_months), 12, 0)
base += np.where(np.isin(order_month, summer_months),  18, 0)
base += np.where(order_month == 12, 8, 0)              # year-end slowdown

# weekend orders slightly slower
base += np.where(order_day_of_week >= 4, 3, 0)

# add realistic noise
noise = np.random.normal(0, base * 0.18)
days_to_arrival = np.clip(base + noise, 7, 180).round().astype(int)

arrival_date = order_date + pd.to_timedelta(days_to_arrival, unit='D')

df = pd.DataFrame({
    'agency':           agency,
    'gender':           gender,
    'country':          country,
    'job':              job,
    'order_month':      order_month,
    'order_day_of_week':order_day_of_week,
    'order_hour':       order_hour,
    'order_date':       order_date.strftime('%Y-%m-%d'),
    'arrival_date':     arrival_date.strftime('%Y-%m-%d'),
    'days_to_arrival':  days_to_arrival,
})

df.to_csv(OUT, index=False)
print(f"Saved {OUT} — {len(df):,} rows")
print(df.groupby('country')['days_to_arrival'].mean().round(1).to_string())
