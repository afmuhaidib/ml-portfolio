import joblib
import pandas as pd

# ── EDIT THESE ────────────────────────────────
MODEL = '../model/home_price_model.pkl'

# single prediction: fill in values matching FEATURES order
# set to None to use CSV batch mode instead
SINGLE = {'sqft': 1800, 'bedrooms': 3, 'bathrooms': 2}

CSV     = '../databases/home_prices.csv'   # used only when SINGLE = None
SEP     = ','
OUTPUT  = '../databases/results.csv'       # set to None to skip saving
# ─────────────────────────────────────────────

bundle   = joblib.load(MODEL)
scaler   = bundle['scaler']
model    = bundle['model']
features = bundle['features']

inputs = pd.DataFrame([SINGLE] if SINGLE else pd.read_csv(CSV, sep=SEP)[features])

inputs['predicted_price'] = model.predict(scaler.transform(inputs[features]))
inputs['predicted_price'] = inputs['predicted_price'].round(2)

print(inputs.to_string(index=False))

if OUTPUT and not SINGLE:
    inputs.to_csv(OUTPUT, index=False)
    print(f"\nSaved to {OUTPUT}")
