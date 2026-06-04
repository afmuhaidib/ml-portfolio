import joblib
import pandas as pd

# ── EDIT THESE ────────────────────────────────
MODEL   = '../model/titanic_model.pkl'
CSV     = '../datasets/test.csv'
SEP     = ','
FEATURE = ['Sex', 'Pclass', 'Age', 'SibSp', 'Parch', 'Embarked']
OUTPUT  = '../datasets/submission.csv'
# ─────────────────────────────────────────────

bundle = joblib.load(MODEL)
model, columns = bundle['model'], bundle['columns']

raw = pd.read_csv(CSV, sep=SEP)
X   = pd.get_dummies(raw[FEATURE]).reindex(columns=columns, fill_value=0)

submission = pd.DataFrame({'PassengerId': raw['PassengerId'], 'Survived': model.predict(X)})
submission.to_csv(OUTPUT, index=False)
print(f"Saved → {OUTPUT}")
