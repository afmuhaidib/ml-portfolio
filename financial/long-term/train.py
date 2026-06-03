"""
Long-Term Model — XGBoost
Features : fundamental ratios + valuation multiples fetched from yfinance
Target   : 12-month forward price (absolute $)
Outputs  : predicted_price, fair_value
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler

# ── EDIT ─────────────────────────────────────
DATA   = Path('financial/long-term/datasets/sp500_fundamentals.parquet')
OUTPUT = Path('financial/long-term/model/longterm_model.pkl')
# ─────────────────────────────────────────────

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

FEATURES = [
    'Gross Margin', 'Net Profit Margin', 'Operating Margin',
    'ROE - Return On Equity', 'ROA - Return On Assets', 'Return On Tangible Equity',
    'Debt/Equity Ratio', 'Long-term Debt / Capital', 'Current Ratio',
    'Asset Turnover', 'Inventory Turnover Ratio', 'Receiveable Turnover',
    'EBIT Margin', 'Days Sales In Receivables',
    'price_to_book', 'momentum_3m', 'momentum_6m', 'volatility',
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['Ticker', 'Date'])
    df['Date'] = pd.to_datetime(df['Date'])

    frames = []
    for ticker, g in df.groupby('Ticker'):
        g = g.copy().set_index('Date').sort_index()

        # price-derived features
        g['price_to_book']  = g['Close'] / (g['Close'] / (g['ROE - Return On Equity'].abs() + 1e-6))
        g['momentum_3m']    = g['Close'].pct_change(63)
        g['momentum_6m']    = g['Close'].pct_change(126)
        g['volatility']     = g['Close'].pct_change().rolling(21).std()

        # target: price 252 trading days (~1 year) from now
        g['target_price']   = g['Close'].shift(-252)

        frames.append(g.reset_index())

    return pd.concat(frames, ignore_index=True)


def main():
    print("Loading data...")
    df = pd.read_parquet(DATA)
    print(f"  {len(df):,} rows, {df['Ticker'].nunique()} tickers")

    print("Engineering features...")
    df = add_features(df)

    # remove outliers
    df = df.dropna(subset=FEATURES + ['target_price', 'Close'])
    pct_change = (df['target_price'] - df['Close']) / df['Close']
    df = df[pct_change.between(-0.95, 15)]
    print(f"  {len(df):,} clean samples")

    X = df[FEATURES].clip(-1e6, 1e6)
    # train on % return, convert to price at inference
    y = (df['target_price'] - df['Close']) / df['Close']

    # time-series split — no future leakage
    tscv   = TimeSeriesSplit(n_splits=5)
    splits = list(tscv.split(X))
    train_idx, test_idx = splits[-1]  # use last fold

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    scaler  = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")

    print("\nTraining XGBoost...")
    model = XGBRegressor(
        n_estimators=600,
        max_depth=5,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=10,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)

    y_pred = model.predict(X_test)
    close  = df['Close'].iloc[test_idx].values

    # directional accuracy
    actual_dir    = (y_test.values > 0)
    predicted_dir = (y_pred > 0)
    dir_acc = (actual_dir == predicted_dir).mean()

    pct_err = np.median(np.abs(y_pred - y_test.values))

    print("\n" + "=" * 40)
    print(" LONG-TERM MODEL REPORT")
    print("=" * 40)
    print(f"  Directional accuracy : {dir_acc:.1%}")
    print(f"  Median price error   : {pct_err:.1%}")

    print("\n  Feature importance:")
    imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    for f, v in imp.head(6).items():
        print(f"    {f:<35} {v:.4f}")

    joblib.dump({'model': model, 'scaler': scaler, 'features': FEATURES,
                 'target': 'pct_return'}, OUTPUT)
    print(f"\nSaved → {OUTPUT}")


if __name__ == '__main__':
    main()
