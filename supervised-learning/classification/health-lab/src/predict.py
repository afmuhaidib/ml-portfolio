import joblib
import pandas as pd

# ── EDIT THESE ────────────────────────────────
MODEL = '../model/health_model.pkl'

LAB = {
    # ── From report (Abdullah, 13M, 02-Jun-2026) ──────────────
    'WBC': 5.24,         'Neutrophils_pct': 31.0,  'Lymphocytes_pct': 55.0,   # NEUT low, LYM high
    'Monocytes_pct': 8.61,'Eosinophils_pct': 4.09, 'RBC': 4.71,
    'Hemoglobin': 13.3,  'Hematocrit': 40.9,       'MCV': 86.8,
    'MCH': 28.1,         'Platelets': 336.0,
    'Glucose': 93.1,     'HbA1c': 4.5,             'Creatinine': 0.9,         # glucose converted from 5.17 mmol/L
    'BUN': 14.0,         'eGFR': 95.0,             'Sodium': 140.0,
    'Potassium': 4.0,    'Chloride': 102.0,         'CO2': 24.0,
    'Calcium': 9.22,     'Magnesium': 2.16,         'Phosphorus': 3.5,         # Ca: 2.30 mmol/L, Mg: 0.89 mmol/L
    'ALT': 33.0,         'AST': 32.0,               'ALP': 72.0,
    'GGT': 22.0,         'Bilirubin': 0.70,         'Albumin': 4.56,           # bili: 12 µmol/L, alb: 45.6 g/L
    'Total_Protein': 7.2,'Total_Cholesterol': 185.6,'LDL': 108.0,              # chol: 4.80 mmol/L
    'HDL': 56.0,         'Triglycerides': 110.0,    'TSH': 1.24,
    'Free_T4': 1.2,      'Iron': 90.0,              'Ferritin': 99.82,         # FERRITIN high (ref 13.7-78.8)
    'TIBC': 330.0,       'Transferrin_Sat': 30.0,   'CRP': 1.2,
    'ESR': 10.0,         'Vitamin_D': 16.27,        'B12': 692.0,              # VitD: 40.6 nmol/L = 16.3 ng/mL (LOW)
    'Folate': 12.0,      'Uric_Acid': 5.5,          'LDH': 155.0,
    'Amylase': 78.0,     'Lipase': 38.0,            'PT_INR': 1.0,
    'AFP': 3.5,          'PSA': 1.2,                'CEA': 1.8,
    'Cortisol': 15.0,    'Testosterone': 400.0,
    'age': 13,           'sex': 'M',
}
THRESHOLD = 0.25    # show diseases with risk above this
# ─────────────────────────────────────────────

bundle   = joblib.load(MODEL)
model    = bundle['model']
features = bundle['features']
diseases = bundle['diseases']

X = pd.get_dummies(pd.DataFrame([LAB]), columns=['sex']).reindex(columns=features, fill_value=0)

probs = [est.predict_proba(X)[0][1] for est in model.estimators_]
risks = pd.DataFrame({'Disease': diseases, 'Risk': probs})
risks = risks[risks['Risk'] >= THRESHOLD].sort_values('Risk', ascending=False)

risks['Level'] = risks['Risk'].apply(
    lambda r: 'HIGH' if r > 0.65 else ('MODERATE' if r > 0.4 else 'LOW')
)
risks['Risk'] = (risks['Risk'] * 100).round(1).astype(str) + '%'

print(f"\n{'='*50}")
print(f"  EARLY RISK REPORT  (threshold: >{THRESHOLD*100:.0f}%)")
print(f"{'='*50}")
print(risks.to_string(index=False))
print(f"\n  Total flagged: {len(risks)} / {len(diseases)} diseases")
