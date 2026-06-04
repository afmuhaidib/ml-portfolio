import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

# ── EDIT THESE ────────────────────────────────
CSV     = '../datasets/workers.csv'
FEATURE = ['agency', 'gender', 'country', 'job',
           'order_month', 'order_day_of_week', 'order_hour',
           'agency_country_avg', 'agency_job_avg']
TARGET  = 'days_to_arrival'
OUTPUT  = '../model/worker_model.pkl'
# ─────────────────────────────────────────────

df = pd.read_csv(CSV)

# engineer agency avg features from historical data
df['agency_country_avg'] = df.groupby(['agency', 'country'])[TARGET].transform('mean')
df['agency_job_avg']     = df.groupby(['agency', 'job'])[TARGET].transform('mean')

# save agency lookup tables for predict.py
agency_country_avg = df.groupby(['agency', 'country'])[TARGET].mean().reset_index()
agency_country_avg.columns = ['agency', 'country', 'agency_country_avg']
agency_job_avg = df.groupby(['agency', 'job'])[TARGET].mean().reset_index()
agency_job_avg.columns = ['agency', 'job', 'agency_job_avg']

X = pd.get_dummies(df[FEATURE])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                     subsample=0.8, n_jobs=-1, random_state=42)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)

mae = mean_absolute_error(y_test, model.predict(X_test))
print(f"\nMAE: {mae:.1f} days")

imp = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print("\nTop features:")
print(imp.head(10).to_string())

joblib.dump({
    'model': model,
    'columns': list(X_train.columns),
    'agency_country_avg': agency_country_avg,
    'agency_job_avg': agency_job_avg,
}, OUTPUT)
print(f"\nSaved → {OUTPUT}")
