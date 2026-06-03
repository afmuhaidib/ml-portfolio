import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── EDIT THESE ────────────────────────────────
CSV      = '../databases/home_prices.csv'
SEP      = ','
FEATURES = ['sqft', 'bedrooms', 'bathrooms']  # input columns
TARGET   = 'price'                             # column to predict
OUTPUT   = '../model/home_price_model.pkl'
# ─────────────────────────────────────────────

df = pd.read_csv(CSV, sep=SEP).dropna(subset=FEATURES + [TARGET])
X  = df[FEATURES]
y  = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n" + "=" * 40)
print(f" PERFORMANCE REPORT: {TARGET}")
print("=" * 40)
print(f"  R²  score : {r2_score(y_test, y_pred):.4f}")
print(f"  MAE       : {mean_absolute_error(y_test, y_pred):,.2f}")
print(f"  RMSE      : {mean_squared_error(y_test, y_pred) ** 0.5:,.2f}")

joblib.dump({'scaler': scaler, 'model': model, 'features': FEATURES}, OUTPUT)
print(f"\nModel saved to {OUTPUT}")
