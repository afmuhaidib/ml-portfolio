import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# ── EDIT THESE ────────────────────────────────
CSV     = '../datasets/train.csv'
SEP     = ','
FEATURE = ['Sex', 'Pclass', 'Age', 'SibSp', 'Parch', 'Embarked']
TARGET  = 'Survived'
OUTPUT  = '../model/titanic_model.pkl'
# ─────────────────────────────────────────────

df = pd.read_csv(CSV, sep=SEP).dropna(subset=FEATURE + [TARGET])
X  = pd.get_dummies(df[FEATURE])
y  = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print(classification_report(y_test, model.predict(X_test)))

joblib.dump({'model': model, 'columns': X_train.columns.tolist()}, OUTPUT)
print(f"Saved → {OUTPUT}")
