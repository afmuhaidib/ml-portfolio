import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

CSV     = '../datasets/diabetes_merged.csv'
FEATURE = ['Gender','Age','BMI','Cholesterol','Glucose','HbA1c',
           'BloodPressure','Insulin','Smoker','Hypertension','HeartDisease',
           'Stroke','PhysActivity','HvyAlcoholConsump','GenHlth',
           'MentHlth','PhysHlth','DiffWalk']
TARGET  = 'Diabetes'
OUTPUT  = '../model/diabetes-model.pkl'

df = pd.read_csv(CSV).dropna(subset=FEATURE + [TARGET])
X  = df[FEATURE]
y  = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1, class_weight='balanced')
model.fit(X_train, y_train)

print(classification_report(y_test, model.predict(X_test)))

joblib.dump({'model': model, 'features': FEATURE}, OUTPUT)
print(f"Saved → {OUTPUT}")
