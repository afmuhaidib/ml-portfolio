import joblib
import pandas as pd

# ── EDIT THESE ────────────────────────────────
MODEL   = '../model/diabetes-model.pkl'
ROW     = [1, 0, 27.1, 1, 0, 0, 1, 0, 3, 2, 3, 0, 0, 7]   # single prediction, or set to None to use CSV
CSV     = None
SEP     = ','
FEATURE = ['HighBP', 'HighChol', 'BMI', 'Smoker', 'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'HvyAlcoholConsump', 'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age']
TARGET  = 'Diabetes_012'   # set to None if CSV has no labels
OUTPUT  = None
# ─────────────────────────────────────────────
model = joblib.load(MODEL)['model']

X = pd.DataFrame([ROW], columns=FEATURE) if ROW else pd.read_csv(CSV, sep=SEP)[FEATURE]

df = X.copy()
df['prediction'] = model.predict(X)
df['confidence'] = model.predict_proba(X).max(axis=1).round(4)

if not ROW and CSV and TARGET:
    actual = pd.read_csv(CSV, sep=SEP)[TARGET]
    df['correct'] = (df['prediction'].values == actual.values)

print(df.to_string(index=False))

if OUTPUT:
    df.to_csv(OUTPUT, index=False)
    print(f"\nSaved → {OUTPUT}")
