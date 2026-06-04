import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# ── EDIT THESE ────────────────────────────────
CSV     = '../datasets/diabetes.csv'
SEP     = ','
FEATURE = ['HighBP', 'HighChol', 'BMI', 'Smoker', 'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'HvyAlcoholConsump', 'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age']
TARGET  = 'Diabetes_012'
OUTPUT  = '../model/diabetes-model.pkl'
# ─────────────────────────────────────────────

df = pd.read_csv(CSV, sep=SEP).dropna(subset=FEATURE + [TARGET])
X  = df[FEATURE]
y  = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

print(classification_report(y_test, model.predict(X_test)))

joblib.dump({'model': model, 'features': FEATURE}, OUTPUT)
print(f"Saved → {OUTPUT}")
