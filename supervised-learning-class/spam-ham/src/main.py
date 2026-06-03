import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# ── EDIT THESE ────────────────────────────────
CSV      = '../databases/spamhamdata.csv'
SEP      = '\t'          # ',' for comma, '\t' for tab
FEATURE  = 'email'       # input column
TARGET   = 'tag'         # label column
OUTPUT   = '../model/spam_model.pkl'
# ─────────────────────────────────────────────

df = pd.read_csv(CSV, sep=SEP).dropna(subset=[FEATURE, TARGET])
X  = df[FEATURE]
y  = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test  = vectorizer.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n" + "=" * 40)
print(f" PERFORMANCE REPORT: {TARGET}")
print("=" * 40)
print(classification_report(y_test, y_pred))

joblib.dump({'vectorizer': vectorizer, 'model': model}, OUTPUT)
print(f"Model saved to {OUTPUT}")
