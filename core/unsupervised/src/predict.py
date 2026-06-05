import joblib
import pandas as pd

MODEL = '../model/customers.pkl'

# --- set CSV path to predict from file, or leave "" to use CUSTOMER dict below ---
CSV_PATH = ""
# ----------------------------------------------------------------------------------

# --- edit single query here ---
CUSTOMER = {
    'age':           35,
    'tenure':        12,
    'monthly_spend': 80.0,
    'num_products':  2,
    'support_calls': 1,
    'has_contract':  1,
}
# ------------------------------

bundle  = joblib.load(MODEL)
features = bundle['features']

if CSV_PATH:
    df = pd.read_csv(CSV_PATH)
    X  = bundle['scaler'].transform(df[features])
    df['cluster'] = bundle['model'].predict(X)
    print(df[features + ['cluster']].to_string(index=False))
else:
    X       = bundle['scaler'].transform(pd.DataFrame([CUSTOMER]))
    cluster = bundle['model'].predict(X)[0]
    print(f"\nCustomer → Cluster {cluster}")
