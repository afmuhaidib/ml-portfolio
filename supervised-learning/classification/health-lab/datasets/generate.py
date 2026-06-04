"""
Generates 300,000 synthetic patients with 50 lab features and 100 disease labels.
Each disease is defined by which labs it elevates/lowers and its prevalence.
Run once: uv run generate.py
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N   = 300_000
OUT = 'synthetic_patients.csv'

# ── Lab baselines: (mean, std, min_clip, max_clip) ──────────────────────────
LABS = {
    # CBC
    'WBC':              (7.0,   2.0,   1.5,  30.0),
    'Neutrophils_pct':  (62.0,  10.0, 20.0,  90.0),
    'Lymphocytes_pct':  (28.0,   8.0,  5.0,  60.0),
    'Monocytes_pct':    (6.0,    2.0,  1.0,  15.0),
    'Eosinophils_pct':  (2.5,    2.0,  0.0,  20.0),
    'RBC':              (4.8,    0.5,  2.0,   7.0),
    'Hemoglobin':       (14.0,   1.5,  5.0,  20.0),
    'Hematocrit':       (42.0,   4.0, 15.0,  60.0),
    'MCV':              (90.0,   7.0, 60.0, 120.0),
    'MCH':              (30.0,   3.0, 18.0,  40.0),
    'Platelets':        (250.0, 65.0, 20.0, 900.0),
    # CMP
    'Glucose':          (90.0,  12.0, 45.0, 500.0),
    'HbA1c':            (5.2,    0.4,  3.5,  14.0),
    'Creatinine':       (0.9,    0.2,  0.3,   9.0),
    'BUN':              (14.0,   4.0,  3.0,  90.0),
    'eGFR':             (95.0,  18.0,  4.0, 120.0),
    'Sodium':           (140.0,  2.5,118.0, 158.0),
    'Potassium':        (4.0,    0.4,  2.2,   6.8),
    'Chloride':         (102.0,  2.5, 88.0, 116.0),
    'CO2':              (24.0,   2.0,  8.0,  35.0),
    'Calcium':          (9.5,    0.4,  6.5,  13.5),
    'Magnesium':        (2.0,    0.25, 0.8,   3.5),
    'Phosphorus':       (3.5,    0.5,  1.0,   7.0),
    'ALT':              (20.0,  12.0,  3.0, 600.0),
    'AST':              (20.0,  10.0,  3.0, 600.0),
    'ALP':              (72.0,  25.0, 15.0, 600.0),
    'GGT':              (22.0,  15.0,  3.0, 600.0),
    'Bilirubin':        (0.7,    0.3,  0.1,  18.0),
    'Albumin':          (4.3,    0.3,  1.2,   5.5),
    'Total_Protein':    (7.2,    0.4,  3.5,   9.5),
    # Lipids
    'Total_Cholesterol':(185.0, 32.0, 80.0, 450.0),
    'LDL':              (108.0, 28.0, 25.0, 350.0),
    'HDL':              (56.0,  14.0, 15.0, 110.0),
    'Triglycerides':    (110.0, 55.0, 20.0,1000.0),
    # Thyroid
    'TSH':              (2.0,    1.2,  0.05, 25.0),
    'Free_T4':          (1.2,    0.18, 0.3,   3.0),
    # Iron
    'Iron':             (90.0,  28.0, 15.0, 280.0),
    'Ferritin':         (75.0,  60.0,  2.0, 600.0),
    'TIBC':             (330.0, 45.0,150.0, 550.0),
    'Transferrin_Sat':  (30.0,   9.0,  2.0,  80.0),
    # Inflammation
    'CRP':              (1.2,    1.5,  0.1,  80.0),
    'ESR':              (10.0,   8.0,  1.0, 120.0),
    # Vitamins
    'Vitamin_D':        (32.0,  12.0,  4.0,  90.0),
    'B12':              (500.0,180.0, 80.0,1600.0),
    'Folate':           (12.0,   4.5,  1.5,  30.0),
    # Other
    'Uric_Acid':        (5.5,    1.3,  1.5,  12.0),
    'LDH':              (155.0, 38.0, 60.0, 600.0),
    'Amylase':          (78.0,  28.0, 10.0, 500.0),
    'Lipase':           (38.0,  18.0,  5.0, 400.0),
    'PT_INR':           (1.0,   0.08, 0.75,   5.0),
    'AFP':              (3.5,    2.5,  0.5, 250.0),
    'PSA':              (1.2,    1.2,  0.1,  25.0),
    'CEA':              (1.8,    1.2,  0.5,  25.0),
    'Cortisol':         (15.0,   4.5,  1.0,  55.0),
    'Testosterone':     (400.0,140.0, 20.0,1100.0),
}

FEATURES = list(LABS.keys())

# ── 100 Diseases ────────────────────────────────────────────────────────────
# Format: (name, prevalence, age_min, sex[None/M/F], {lab: delta_mean})
# delta_mean is ADDED to the patient's lab value (negative = lower the lab)
DISEASES = [
    # ── METABOLIC ────────────────────────────────────────────────────────────
    ('Type_2_Diabetes',          0.11, 35, None, {'Glucose': +65,  'HbA1c': +1.9,  'Triglycerides': +90,  'HDL': -12, 'LDL': +15}),
    ('Prediabetes',              0.34, 20, None, {'Glucose': +22,  'HbA1c': +0.8,  'Triglycerides': +30}),
    ('Metabolic_Syndrome',       0.33, 30, None, {'Glucose': +18,  'Triglycerides': +80, 'HDL': -14, 'LDL': +20, 'CRP': +2.0}),
    ('Hypothyroidism',           0.05,  0, None, {'TSH': +8.0,     'Free_T4': -0.4, 'Total_Cholesterol': +30, 'LDL': +20, 'Triglycerides': +20}),
    ('Hyperthyroidism',          0.013, 0, None, {'TSH': -1.7,     'Free_T4': +0.7, 'Total_Cholesterol': -20, 'LDH': +30}),
    ('Hashimotos_Thyroiditis',   0.05,  0, 'F',  {'TSH': +5.0,     'Free_T4': -0.3, 'CRP': +1.5}),
    ('Cushings_Syndrome',        0.002,10, None, {'Cortisol': +20, 'Glucose': +30,  'Potassium': -0.5, 'Albumin': -0.4}),
    ('Addisons_Disease',         0.001, 0, None, {'Cortisol': -10, 'Sodium': -8,    'Potassium': +0.8, 'Glucose': -15}),
    ('Gout',                     0.04, 30, None, {'Uric_Acid': +3.5,'Creatinine': +0.3, 'Triglycerides': +40}),
    ('Hyperparathyroidism',      0.03, 40, None, {'Calcium': +1.5,  'Phosphorus': -0.8, 'ALP': +40}),
    ('Hypoparathyroidism',       0.01,  0, None, {'Calcium': -2.0,  'Phosphorus': +1.0, 'Magnesium': -0.3}),
    ('PCOS',                     0.10,  0, 'F',  {'Testosterone': +80,'Glucose': +15,  'Triglycerides': +35, 'HDL': -8, 'LH_proxy': +0}),
    ('Hyperaldosteronism',       0.01, 30, None, {'Potassium': -0.9, 'Sodium': +4.0}),
    ('Vitamin_D_Deficiency',     0.42,  0, None, {'Vitamin_D': -20,  'Calcium': -0.4, 'ALP': +20}),
    ('Obesity_Metabolic',        0.42, 18, None, {'Glucose': +10,   'Triglycerides': +50,'HDL': -10, 'CRP': +3.0, 'ALT': +15}),

    # ── CARDIOVASCULAR ───────────────────────────────────────────────────────
    ('Hypertension_Risk',        0.47, 30, None, {'Sodium': +3.0,   'Creatinine': +0.15,'LDL': +10,  'CRP': +1.5}),
    ('Coronary_Artery_Disease',  0.08, 40, None, {'LDL': +45,       'CRP': +4.0,   'Triglycerides': +60, 'HDL': -10}),
    ('Congestive_Heart_Failure', 0.025,50, None, {'BUN': +15,       'Creatinine': +0.4, 'Sodium': -5, 'Albumin': -0.5}),
    ('MI_Risk',                  0.04, 40, None, {'LDL': +40,       'CRP': +5.0,   'LDH': +50,  'AST': +20}),
    ('Atherosclerosis',          0.10, 40, None, {'LDL': +35,       'Total_Cholesterol': +40, 'CRP': +2.5, 'HDL': -8}),
    ('Atrial_Fibrillation_Risk', 0.04, 50, None, {'Potassium': -0.5,'Magnesium': -0.3,  'TSH': +2.0}),
    ('DVT_Risk',                 0.01, 30, None, {'PT_INR': +0.4,   'CRP': +3.0,   'Albumin': -0.3}),
    ('Hypercholesterolemia',     0.38,  0, None, {'Total_Cholesterol': +65,'LDL': +55}),
    ('Hypertriglyceridemia',     0.25,  0, None, {'Triglycerides': +200,'HDL': -8,  'Glucose': +10}),
    ('Low_HDL_Syndrome',         0.30,  0, None, {'HDL': -22,       'Triglycerides': +45,'LDL': +10}),
    ('Peripheral_Artery_Disease',0.03, 50, None, {'LDL': +30,       'CRP': +3.5,   'Creatinine': +0.3}),
    ('Cardiac_Inflammation',     0.01,  0, None, {'CRP': +8.0,      'WBC': +3.0,   'ESR': +25, 'LDH': +40}),

    # ── KIDNEY ───────────────────────────────────────────────────────────────
    ('CKD_Stage_1',              0.08, 30, None, {'eGFR': -10,      'Creatinine': +0.2}),
    ('CKD_Stage_2',              0.07, 40, None, {'eGFR': -30,      'Creatinine': +0.5, 'BUN': +8}),
    ('CKD_Stage_3',              0.05, 50, None, {'eGFR': -50,      'Creatinine': +1.2, 'BUN': +20, 'Phosphorus': +0.8, 'Potassium': +0.5}),
    ('CKD_Stage_4',              0.01, 50, None, {'eGFR': -70,      'Creatinine': +3.0, 'BUN': +40, 'Calcium': -0.8, 'Hemoglobin': -2.5}),
    ('CKD_Stage_5_ESRD',         0.002,50, None, {'eGFR': -88,      'Creatinine': +7.0, 'BUN': +60, 'Potassium': +1.2, 'Hemoglobin': -4.0}),
    ('Nephrotic_Syndrome',       0.005, 0, None, {'Albumin': -1.5,  'Total_Cholesterol': +60,'Triglycerides': +100, 'Creatinine': +0.5}),
    ('Kidney_Stones',            0.11, 20, None, {'Uric_Acid': +2.0,'Calcium': +0.5,    'Creatinine': +0.1}),
    ('Acute_Kidney_Injury',      0.005, 0, None, {'Creatinine': +2.5,'BUN': +35,        'eGFR': -55, 'Potassium': +0.8}),

    # ── LIVER ────────────────────────────────────────────────────────────────
    ('NAFLD',                    0.25, 20, None, {'ALT': +35,       'AST': +20,    'GGT': +25, 'Triglycerides': +55, 'Glucose': +12}),
    ('Alcoholic_Liver_Disease',  0.05, 25, None, {'GGT': +80,       'AST': +60,    'ALT': +30, 'Bilirubin': +0.8, 'MCV': +8}),
    ('Viral_Hepatitis',          0.01,  0, None, {'ALT': +180,      'AST': +150,   'Bilirubin': +2.5, 'ALP': +60}),
    ('Liver_Cirrhosis',          0.01, 40, None, {'Albumin': -1.2,  'PT_INR': +0.8,'Bilirubin': +2.0, 'Platelets': -80, 'AST': +40}),
    ('Cholestasis',              0.01,  0, None, {'ALP': +180,      'GGT': +120,   'Bilirubin': +3.5, 'Total_Cholesterol': +30}),
    ('Hemochromatosis',          0.003, 0, None, {'Iron': +80,      'Ferritin': +300,'Transferrin_Sat': +30, 'ALT': +25, 'AST': +20}),
    ('Gilberts_Syndrome',        0.08,  0, None, {'Bilirubin': +0.9, 'ALT': 0,     'AST': 0}),

    # ── BLOOD / HEMATOLOGY ───────────────────────────────────────────────────
    ('Iron_Deficiency_Anemia',   0.08,  0, None, {'Hemoglobin': -3.5,'RBC': -0.7,  'MCV': -12, 'MCH': -5, 'Iron': -45, 'Ferritin': -50, 'TIBC': +80, 'Transferrin_Sat': -18}),
    ('B12_Deficiency_Anemia',    0.06, 40, None, {'B12': -320,      'Hemoglobin': -2.5,'MCV': +14, 'RBC': -0.5}),
    ('Folate_Deficiency_Anemia', 0.04,  0, None, {'Folate': -8.0,   'Hemoglobin': -2.0,'MCV': +12}),
    ('Aplastic_Anemia',          0.001, 0, None, {'WBC': -4.0,      'RBC': -1.5,   'Platelets': -150,'Hemoglobin': -5.0}),
    ('Polycythemia_Vera',        0.002,50, None, {'RBC': +1.8,      'Hemoglobin': +3.5,'Hematocrit': +10, 'Platelets': +200, 'WBC': +4.0}),
    ('Thrombocytopenia',         0.03,  0, None, {'Platelets': -140}),
    ('Thrombocytosis',           0.02,  0, None, {'Platelets': +280, 'CRP': +2.0}),
    ('Leukopenia',               0.01,  0, None, {'WBC': -3.5,      'Neutrophils_pct': -15}),
    ('Leukocytosis',             0.05,  0, None, {'WBC': +8.0,      'CRP': +4.0}),
    ('Neutropenia',              0.01,  0, None, {'Neutrophils_pct': -30,'WBC': -2.0}),
    ('Eosinophilia',             0.03,  0, None, {'Eosinophils_pct': +8.0,'WBC': +1.5}),
    ('Hemolytic_Anemia',         0.005, 0, None, {'LDH': +120,      'Bilirubin': +1.5,'Hemoglobin': -3.0,'RBC': -0.8, 'Hematocrit': -8}),

    # ── INFECTION / INFLAMMATION ─────────────────────────────────────────────
    ('Bacterial_Infection',      0.15,  0, None, {'WBC': +6.0,      'Neutrophils_pct': +15,'CRP': +12.0, 'ESR': +30}),
    ('Viral_Infection',          0.20,  0, None, {'Lymphocytes_pct': +15,'WBC': +1.5,  'CRP': +3.0, 'ESR': +15}),
    ('Sepsis_Risk',              0.005, 0, None, {'WBC': +12.0,     'CRP': +25.0,  'Lactate_proxy': 0, 'Albumin': -0.8, 'Platelets': -80}),
    ('Chronic_Inflammation',     0.08, 30, None, {'CRP': +5.0,      'ESR': +30,    'Albumin': -0.4, 'Hemoglobin': -1.2}),
    ('Parasitic_Infection',      0.03,  0, None, {'Eosinophils_pct': +10.0,'IgE_proxy': 0, 'Hemoglobin': -1.5}),
    ('Autoimmune_Activity',      0.05,  0, 'F',  {'ESR': +35,       'CRP': +4.0,   'Albumin': -0.3, 'WBC': -1.5, 'Platelets': -40}),
    ('HIV_Immunodeficiency_Risk',0.004, 0, None, {'Lymphocytes_pct': -15,'WBC': -2.0,'Albumin': -0.5, 'CRP': +2.0}),

    # ── BONE / MINERAL ───────────────────────────────────────────────────────
    ('Osteoporosis_Risk',        0.10, 50, None, {'Vitamin_D': -15, 'Calcium': -0.3, 'ALP': +30, 'Phosphorus': -0.3}),
    ('Rickets_Osteomalacia',     0.01,  0, None, {'Vitamin_D': -25, 'Calcium': -0.8, 'Phosphorus': -0.8, 'ALP': +80}),
    ('Pagets_Disease',           0.003,55, None, {'ALP': +200,      'Calcium': 0}),
    ('Hypercalcemia',            0.01, 40, None, {'Calcium': +2.0,  'Phosphorus': -0.5,'BUN': +5}),
    ('Hypocalcemia',             0.01,  0, None, {'Calcium': -1.8,  'Albumin': -0.4}),

    # ── HORMONAL / ENDOCRINE ─────────────────────────────────────────────────
    ('Male_Hypogonadism',        0.04,  0, 'M',  {'Testosterone': -220,'LH_proxy': 0, 'Hemoglobin': -1.0}),
    ('Female_Hypogonadism',      0.01,  0, 'F',  {'Testosterone': -30, 'FSH_proxy': 0, 'Calcium': -0.2}),
    ('Adrenal_Insufficiency',    0.002, 0, None, {'Cortisol': -10,  'Sodium': -6,   'Potassium': +0.6, 'Glucose': -12, 'Albumin': -0.3}),
    ('Acromegaly',               0.0006,20,None, {'Glucose': +20,   'Phosphorus': +0.5,'Triglycerides': +30}),

    # ── CANCER RISK MARKERS ──────────────────────────────────────────────────
    ('Prostate_Cancer_Risk',     0.04, 50, 'M',  {'PSA': +8.0,      'ALP': +30}),
    ('Colorectal_Cancer_Risk',   0.02, 45, None, {'CEA': +8.0,      'Hemoglobin': -2.0,'Iron': -20, 'Ferritin': -20}),
    ('Liver_Cancer_Risk',        0.003,50, None, {'AFP': +80.0,     'ALT': +60,    'AST': +50, 'ALP': +80, 'Bilirubin': +1.5}),
    ('Pancreatic_Cancer_Risk',   0.002,55, None, {'Glucose': +35,   'Amylase': +50,'Lipase': +60, 'Albumin': -0.5, 'CEA': +5}),
    ('Leukemia_Risk',            0.001, 0, None, {'WBC': +35.0,     'LDH': +150,   'Uric_Acid': +2.5, 'Hemoglobin': -3.0, 'Platelets': -80}),
    ('Lymphoma_Risk',            0.002,20, None, {'LDH': +100,      'ESR': +45,    'Albumin': -0.6, 'WBC': +3.0}),
    ('Multiple_Myeloma_Risk',    0.001,60, None, {'Total_Protein': +1.8,'Calcium': +1.2,'Albumin': -0.8, 'Creatinine': +0.8, 'ESR': +60}),
    ('Thyroid_Cancer_Risk',      0.002, 0, None, {'TSH': -1.5,      'ALP': +25,    'Calcium': +0.3}),
    ('Ovarian_Cancer_Risk',      0.01, 40, 'F',  {'CEA': +4.0,      'CRP': +3.0,   'Albumin': -0.3}),
    ('Lung_Cancer_Risk',         0.01, 50, None, {'CEA': +5.0,      'LDH': +60,    'Albumin': -0.4, 'Calcium': +0.3, 'CRP': +4.0}),

    # ── NUTRITIONAL DEFICIENCIES ─────────────────────────────────────────────
    ('Iron_Deficiency',          0.12,  0, None, {'Iron': -40,      'Ferritin': -45,'TIBC': +70, 'Transferrin_Sat': -15}),
    ('Zinc_Deficiency',          0.07,  0, None, {'Albumin': -0.3,  'ALP': -15,    'Testosterone': -50}),
    ('Magnesium_Deficiency',     0.10,  0, None, {'Magnesium': -0.5,'Potassium': -0.4,'Calcium': -0.3}),
    ('Protein_Malnutrition',     0.03,  0, None, {'Albumin': -1.0,  'Total_Protein': -1.2,'Hemoglobin': -1.5, 'Lymphocytes_pct': -8}),
    ('Vitamin_K_Deficiency',     0.03,  0, None, {'PT_INR': +0.8,   'Albumin': -0.2}),
    ('Selenium_Deficiency',      0.05,  0, None, {'Free_T4': -0.1,  'TSH': +0.5}),

    # ── ELECTROLYTE DISORDERS ────────────────────────────────────────────────
    ('Hyponatremia',             0.03,  0, None, {'Sodium': -9.0}),
    ('Hypernatremia',            0.01,  0, None, {'Sodium': +8.0}),
    ('Hypokalemia',              0.03,  0, None, {'Potassium': -0.9,'Magnesium': -0.3}),
    ('Hyperkalemia',             0.02,  0, None, {'Potassium': +1.2, 'Creatinine': +0.3}),
    ('Hypomagnesemia',           0.04,  0, None, {'Magnesium': -0.6,'Potassium': -0.3, 'Calcium': -0.2}),
    ('Metabolic_Acidosis',       0.02,  0, None, {'CO2': -6.0,      'Creatinine': +0.4}),
    ('Metabolic_Alkalosis',      0.01,  0, None, {'CO2': +7.0,      'Potassium': -0.5}),

    # ── DIGESTIVE / OTHER ────────────────────────────────────────────────────
    ('Pancreatitis_Risk',        0.03, 20, None, {'Amylase': +150,  'Lipase': +200, 'Triglycerides': +200, 'CRP': +8.0, 'WBC': +4.0}),
    ('Celiac_Disease',           0.01,  0, None, {'Folate': -5.0,   'Iron': -30,    'Calcium': -0.5, 'Albumin': -0.4, 'Hemoglobin': -1.8}),
    ('IBD_Risk',                 0.01,  0, None, {'CRP': +6.0,      'ESR': +30,     'Albumin': -0.5, 'Hemoglobin': -1.5, 'Platelets': +80}),

    # ── AUTOIMMUNE ───────────────────────────────────────────────────────────
    ('Rheumatoid_Arthritis_Risk',0.01,  0, 'F',  {'CRP': +7.0,      'ESR': +40,     'Hemoglobin': -1.2, 'Albumin': -0.3, 'Platelets': +60}),
    ('Lupus_Risk',               0.005, 0, 'F',  {'WBC': -2.5,      'Platelets': -60,'ESR': +45,  'CRP': +3.0, 'Albumin': -0.4, 'Creatinine': +0.3}),
    ('Type_1_Diabetes_Risk',     0.005, 0, None, {'Glucose': +45,   'HbA1c': +1.5,  'Amylase': -15}),
    ('Vasculitis_Risk',          0.002, 0, None, {'CRP': +9.0,      'ESR': +50,     'Creatinine': +0.5, 'WBC': +3.0}),
]

DISEASE_NAMES = [d[0] for d in DISEASES]
assert len(DISEASE_NAMES) == 100, f"Expected 100 diseases, got {len(DISEASE_NAMES)}"


def generate():
    print(f"Generating {N:,} synthetic patients...")

    # demographics
    age = np.random.randint(18, 85, N).astype(float)
    sex = np.random.choice(['M', 'F'], N)

    # baseline lab values
    data = {}
    for lab, (mean, std, lo, hi) in LABS.items():
        data[lab] = np.clip(np.random.normal(mean, std, N), lo, hi)

    # sex adjustments
    female = (sex == 'F')
    data['Hemoglobin'][female]   -= 1.3
    data['Hematocrit'][female]   -= 4.0
    data['RBC'][female]          -= 0.4
    data['Testosterone'][female] /= 10   # females have ~10x lower testosterone
    data['Iron'][female]         -= 8
    data['Ferritin'][female]     -= 30

    # age adjustments
    age_factor = (age - 18) / 67
    data['Creatinine']       += age_factor * 0.2
    data['eGFR']             -= age_factor * 20
    data['Total_Cholesterol']+= age_factor * 20
    data['LDL']              += age_factor * 15
    data['Glucose']          += age_factor * 8
    data['PSA']              += age_factor * 1.5
    data['BUN']              += age_factor * 5

    df = pd.DataFrame(data)
    df['age'] = age
    df['sex'] = sex

    # ── Apply disease patterns ────────────────────────────────────────────────
    labels = pd.DataFrame(0, index=range(N), columns=DISEASE_NAMES)

    for name, prevalence, age_min, sex_filter, lab_effects in DISEASES:
        # eligible patients
        eligible = (age >= age_min)
        if sex_filter == 'M':
            eligible &= (sex == 'M')
        elif sex_filter == 'F':
            eligible &= (sex == 'F')

        n_eligible = eligible.sum()
        n_cases    = int(n_eligible * prevalence)
        candidates = np.where(eligible)[0]
        chosen     = np.random.choice(candidates, size=min(n_cases, len(candidates)), replace=False)

        labels.loc[chosen, name] = 1

        # shift labs for affected patients
        for lab, delta in lab_effects.items():
            if lab in df.columns:
                noise = np.random.normal(delta, abs(delta) * 0.25 + 1, len(chosen))
                df.loc[chosen, lab] += noise

    # re-clip after disease modifications
    for lab, (mean, std, lo, hi) in LABS.items():
        df[lab] = df[lab].clip(lo, hi)

    # combine
    out = pd.concat([df[FEATURES + ['age', 'sex']], labels], axis=1)
    out.to_csv(OUT, index=False)
    print(f"Saved {OUT} — {len(out):,} rows × {len(out.columns)} columns")
    print(f"Disease prevalence sample:")
    for col in DISEASE_NAMES[:5]:
        print(f"  {col}: {labels[col].sum():,} cases")


if __name__ == '__main__':
    generate()
