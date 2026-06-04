import joblib
import pandas as pd
from datetime import datetime, timedelta

# ── EDIT THESE ────────────────────────────────
MODEL = '../model/worker_model.pkl'

ORDER = {
    'agency':            'Al Majd',
    'gender':            'F',
    'country':           'Philippines',
    'job':               'housemaid',
    'order_month':       datetime.today().month,
    'order_day_of_week': datetime.today().weekday(),
    'order_hour':        datetime.today().hour,
}
# ─────────────────────────────────────────────

bundle             = joblib.load(MODEL)
model              = bundle['model']
columns            = bundle['columns']
agency_country_avg = bundle['agency_country_avg']
agency_job_avg     = bundle['agency_job_avg']

row = ORDER.copy()

# look up agency historical averages
ac = agency_country_avg[
    (agency_country_avg['agency']  == row['agency']) &
    (agency_country_avg['country'] == row['country'])
]
aj = agency_job_avg[
    (agency_job_avg['agency'] == row['agency']) &
    (agency_job_avg['job']    == row['job'])
]
row['agency_country_avg'] = ac['agency_country_avg'].values[0] if len(ac) else 50.0
row['agency_job_avg']     = aj['agency_job_avg'].values[0]     if len(aj) else 50.0

X = pd.get_dummies(pd.DataFrame([row])).reindex(columns=columns, fill_value=0)

days    = round(model.predict(X)[0])
arrival = datetime.today() + timedelta(days=days)

print(f"\n{'='*42}")
print(f"  Worker Arrival Estimate")
print(f"{'='*42}")
print(f"  Agency          : {ORDER['agency']}")
print(f"  Job             : {ORDER['job']} ({ORDER['country']})")
print(f"  Agency avg (country/job): {row['agency_country_avg']:.0f} / {row['agency_job_avg']:.0f} days")
print(f"  Order date      : {datetime.today().strftime('%d %b %Y')}")
print(f"  Estimated arrival: {arrival.strftime('%d %b %Y')}  (~{days} days)")
print(f"{'='*42}")
