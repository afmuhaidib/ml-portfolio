import joblib
import pandas as pd

# ── EDIT THESE ────────────────────────────────
MODEL   = '../model/spam_model.pkl'
TEXT    = 'URGENT: Your email has WON $5,000,000.00 Cash! Claim Now!! 🤑'
CSV     = '../databases/spamhamdata.csv'   # used only when TEXT = None
SEP     = '\t'
FEATURE = 'email'
OUTPUT  = '../databases/results.csv'       # set to None to skip saving
# ─────────────────────────────────────────────

bundle      = joblib.load(MODEL)
transformer = bundle['vectorizer']
model       = bundle['model']

inputs = pd.Series([TEXT] if TEXT else pd.read_csv(CSV, sep=SEP)[FEATURE])

df = pd.DataFrame({'input': inputs})
df['label']       = model.predict(transformer.transform(inputs))
df['confidence']  = model.predict_proba(transformer.transform(inputs)).max(axis=1).round(4)

print(df.to_string(index=False))

if OUTPUT and not TEXT:
    df.to_csv(OUTPUT, index=False)
    print(f"\nSaved to {OUTPUT}")
