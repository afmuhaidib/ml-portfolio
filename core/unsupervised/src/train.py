import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

FEATURES = ['age', 'tenure', 'monthly_spend', 'num_products', 'support_calls', 'has_contract']

df      = pd.read_csv('../datasets/customers.csv')
scaler  = StandardScaler()
X       = scaler.fit_transform(df[FEATURES])

model   = KMeans(n_clusters=4, random_state=42)
model.fit(X)

df['cluster'] = model.labels_
print(df.groupby('cluster')[FEATURES].mean().round(1).to_string())
print(f"\nInertia: {model.inertia_:.2f}")

joblib.dump({'model': model, 'scaler': scaler, 'features': FEATURES}, '../model/customers.pkl')
print("Saved → ../model/customers.pkl")
