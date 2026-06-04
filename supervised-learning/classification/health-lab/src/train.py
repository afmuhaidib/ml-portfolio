import joblib
import pandas as pd
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# ── EDIT THESE ────────────────────────────────
CSV    = '../datasets/synthetic_patients.csv'
OUTPUT = '../model/health_model.pkl'
# ─────────────────────────────────────────────

df      = pd.read_csv(CSV)
FEATURES = [c for c in df.columns if c not in df.filter(regex='_Risk$|_Syndrome|Diabetes|Disease|Anemia|Infection|Disorder|Deficiency|Cirrhosis|Hepatitis|Steatosis|Acidosis|Alkalosis|Pancreatitis|Hypo|Hyper|CKD|DVT|IBD|CHF|NAFLD|ESRD|Celiac|Lupus|Gout|Obesity|Atherosclerosis|Polycythemia|Thrombocyt|Leukoc|Leukopen|Neutro|Eosinoph|Hemolytic|Sepsis|Autoimmune|Fungal|HIV|Osteoporosis|Rickets|Pagets|Acromegaly|Adrenal|Gonadism|Prolactin|Cancer|Myeloma|Lymphoma|Leukemia|Sarcoid|Vasculitis|Rheumatoid|Hashimoto|Cushing|Addisons|PCOS|Aldoster|Congenital|Wilsons|Gilberts|Celiac|Type_1|Type_2|Pre').columns]

# simpler: features = all numeric non-label columns
DISEASE_COLS = [c for c in df.columns if c not in ['age', 'sex'] and df[c].nunique() == 2 and df[c].max() == 1]
FEAT_COLS    = [c for c in df.columns if c not in DISEASE_COLS]

X = pd.get_dummies(df[FEAT_COLS], columns=['sex'])
y = df[DISEASE_COLS]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training on {len(X_train):,} patients | {len(DISEASE_COLS)} diseases...")
model = MultiOutputClassifier(
    XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.1,
                  subsample=0.8, n_jobs=-1, random_state=42, verbosity=0),
    n_jobs=-1
)
model.fit(X_train, y_train)

score = (model.predict(X_test) == y_test.values).mean()
print(f"Mean accuracy across all diseases: {score:.1%}")

joblib.dump({'model': model, 'features': list(X_train.columns), 'diseases': DISEASE_COLS}, OUTPUT)
print(f"Saved → {OUTPUT}")
