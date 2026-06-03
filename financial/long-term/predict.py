"""
Long-Term Predictor
Fetches latest fundamentals for a ticker, returns:
  - current_price
  - predicted_price (12-month forward)
  - fair_value      (DCF-lite from fundamentals)
  - signal          (BUY / HOLD / SELL)
"""

import joblib
import numpy as np
import yfinance as yf
from pathlib import Path

MODEL_PATH = Path('financial/long-term/model/longterm_model.pkl')

RATIO_MAP = {
    'Gross Margin':               ('grossMargins',           1),
    'Net Profit Margin':          ('profitMargins',          1),
    'Operating Margin':           ('operatingMargins',       1),
    'ROE - Return On Equity':     ('returnOnEquity',         1),
    'ROA - Return On Assets':     ('returnOnAssets',         1),
    'Return On Tangible Equity':  ('returnOnEquity',         1),   # proxy
    'Debt/Equity Ratio':          ('debtToEquity',         100),   # yf gives %, divide
    'Long-term Debt / Capital':   ('longTermDebtToCapitalization', 1),
    'Current Ratio':              ('currentRatio',           1),
    'Asset Turnover':             ('assetTurnover',          1),
    'Inventory Turnover Ratio':   ('inventoryTurnover',      1),
    'Receiveable Turnover':       ('revenuePerShare',        1),    # proxy
    'EBIT Margin':                ('ebitdaMargins',          1),    # proxy
    'Days Sales In Receivables':  ('daysSalesOutstanding',   1),
}


def fetch_features(ticker: str) -> tuple[dict, float]:
    t    = yf.Ticker(ticker)
    info = t.info
    hist = t.history(period='6mo')

    if hist.empty:
        raise ValueError(f"No price data for {ticker}")

    close = hist['Close']
    current_price = float(close.iloc[-1])

    feats = {}
    for feat, (key, divisor) in RATIO_MAP.items():
        val = info.get(key)
        if val is None:
            feats[feat] = 0.0
        else:
            feats[feat] = float(val) / divisor

    # price-derived features
    roe = feats.get('ROE - Return On Equity', 1e-6) or 1e-6
    feats['price_to_book']  = current_price / (current_price / (abs(roe) + 1e-6))
    feats['momentum_3m']    = float(close.pct_change(63).iloc[-1]) if len(close) >= 63 else 0.0
    feats['momentum_6m']    = float(close.pct_change(126).iloc[-1]) if len(close) >= 126 else 0.0
    feats['volatility']     = float(close.pct_change().rolling(21).std().iloc[-1])

    return feats, current_price


def fair_value_estimate(info: dict, current_price: float) -> float:
    """Forward-earnings based fair value using sector median P/E."""
    fwd_eps = info.get('forwardEps') or info.get('trailingEps') or 0
    if fwd_eps <= 0:
        # fallback: price/sales ratio
        ps = info.get('priceToSalesTrailing12Months') or 3
        rev_per_share = info.get('revenuePerShare') or 0
        if rev_per_share > 0:
            return float(rev_per_share * (ps * 0.8))
        return current_price  # can't estimate
    # use sector P/E if available, else 18x
    sector_pe = 18.0
    sector = (info.get('sector') or '').lower()
    if 'technology' in sector:
        sector_pe = 25.0
    elif 'health' in sector:
        sector_pe = 22.0
    elif 'financial' in sector:
        sector_pe = 13.0
    elif 'energy' in sector:
        sector_pe = 14.0
    elif 'consumer' in sector:
        sector_pe = 20.0
    return float(fwd_eps * sector_pe)


def predict(ticker: str) -> dict:
    bundle = joblib.load(MODEL_PATH)
    model, scaler, features = bundle['model'], bundle['scaler'], bundle['features']

    t    = yf.Ticker(ticker.upper())
    info = t.info

    feats, current_price = fetch_features(ticker)

    X = np.array([[feats.get(f, 0.0) for f in features]])
    X = scaler.transform(X)

    pct_return     = float(model.predict(X)[0])
    predicted_price = round(current_price * (1 + pct_return), 2)
    fv             = round(fair_value_estimate(info, current_price), 2)
    current_price  = round(current_price, 2)

    # signal based on model's 12-month predicted return
    if pct_return > 0.12:
        signal = 'BUY'
    elif pct_return < -0.05:
        signal = 'SELL'
    else:
        signal = 'HOLD'

    return {
        'ticker':          ticker.upper(),
        'current_price':   current_price,
        'predicted_price': predicted_price,
        'fair_value':      fv,
        'signal':          signal,
        'upside_pct':      round(pct_return * 100, 1),
    }


if __name__ == '__main__':
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else 'AAPL'
    r = predict(ticker)
    print(f"\n{'='*40}")
    print(f"  {r['ticker']}")
    print(f"{'='*40}")
    print(f"  Current price    : ${r['current_price']}")
    print(f"  Predicted (12mo) : ${r['predicted_price']}")
    print(f"  Fair value       : ${r['fair_value']}")
    print(f"  Signal           : {r['signal']}  ({r['upside_pct']:+.1f}%)")
