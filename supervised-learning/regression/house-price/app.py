import os
import uuid
import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER  = 'model'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER,  exist_ok=True)

session = {}  # holds current csv path + trained model key

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file provided'}), 400

    path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}_{file.filename}")
    file.save(path)

    try:
        sep = '\t' if file.filename.endswith('.tsv') else ','
        df  = pd.read_csv(path, sep=sep, nrows=5)
        session['csv_path'] = path
        session['sep']      = sep
        return jsonify({'columns': list(df.columns), 'preview': df.head(3).to_dict(orient='records')})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/train', methods=['POST'])
def train():
    data     = request.json
    features = data.get('features', [])
    target   = data.get('target')
    path     = session.get('csv_path')
    sep      = session.get('sep', ',')

    if not path or not features or not target:
        return jsonify({'error': 'Missing data'}), 400

    df = pd.read_csv(path, sep=sep).dropna(subset=features + [target])
    X  = df[features]
    y  = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    model_path = os.path.join(MODEL_FOLDER, 'current_model.pkl')
    joblib.dump({'scaler': scaler, 'model': model, 'features': features}, model_path)
    session['model_path'] = model_path

    return jsonify({
        'r2':   round(r2_score(y_test, y_pred), 4),
        'mae':  round(mean_absolute_error(y_test, y_pred), 2),
        'rmse': round(mean_squared_error(y_test, y_pred) ** 0.5, 2),
        'rows': len(df),
    })

@app.route('/predict', methods=['POST'])
def predict():
    data       = request.json
    model_path = session.get('model_path')

    if not model_path or not os.path.exists(model_path):
        return jsonify({'error': 'No trained model found'}), 400

    bundle   = joblib.load(model_path)
    features = bundle['features']
    scaler   = bundle['scaler']
    model    = bundle['model']

    try:
        values = [[float(data[f]) for f in features]]
        scaled = scaler.transform(values)
        price  = model.predict(scaled)[0]
        return jsonify({'prediction': round(price, 2)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5001)
