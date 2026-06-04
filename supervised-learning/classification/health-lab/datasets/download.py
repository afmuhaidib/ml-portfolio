"""
Downloads and merges NHANES 2017-2018 lab panels into one dataset.
Run once: uv run download.py
"""

import pandas as pd

BASE = 'https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018'

FILES = ['CBC_J.XPT', 'BIOPRO_J.XPT', 'TCHOL_J.XPT', 'GHB_J.XPT',
         'DIQ_J.XPT', 'MCQ_J.XPT', 'KIQ_U_J.XPT']

RENAME = {
    'LBXWBCSI': 'WBC',            'LBXRBCSI': 'RBC',
    'LBXHGB':   'Hemoglobin',     'LBXPLTSI': 'Platelets',
    'LBXSGL':   'Glucose',        'LBXSCR':   'Creatinine',
    'LBXSBU':   'BUN',            'LBXSNASI': 'Sodium',
    'LBXSKSI':  'Potassium',      'LBXSCA':   'Calcium',
    'LBXSATSI': 'ALT',            'LBXSASSI': 'AST',
    'LBXTC':    'Total_Cholesterol',
    'LBXGH':    'HbA1c',
    'DIQ010':   'diabetes',
    'MCQ160B':  'CHF',
    'MCQ160L':  'liver_condition',
    'KIQ022':   'kidney_condition',
}

print("Downloading NHANES 2017-2018...")
frames = {}
for f in FILES:
    print(f"  {f}...", end=' ', flush=True)
    try:
        df = pd.read_sas(f"{BASE}/{f}", format='xport', encoding='utf-8').set_index('SEQN')
        frames[f] = df
        print(f"{len(df):,} rows")
    except Exception as e:
        print(f"FAILED: {e}")

merged = pd.concat(frames.values(), axis=1, join='inner')
df = merged[[c for c in RENAME if c in merged.columns]].rename(columns=RENAME)

for col in ['diabetes', 'CHF', 'liver_condition', 'kidney_condition']:
    if col in df.columns:
        df[col] = (df[col] == 1).astype(int)

df.dropna(thresh=10, inplace=True)
df.to_csv('nhanes_labs.csv', index=False)
print(f"\nSaved nhanes_labs.csv — {len(df):,} rows × {len(df.columns)} columns")
print(list(df.columns))
