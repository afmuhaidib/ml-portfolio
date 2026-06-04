import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

bundle   = joblib.load('model/diabetes-model.pkl')
model    = bundle['model']
FEATURES = bundle['features']

LABELS = {0: 'No Diabetes', 1: 'Pre-Diabetes', 2: 'Diabetes'}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    d = request.get_json()
    try:
        row = [float(d[f]) for f in FEATURES]
    except (KeyError, ValueError):
        return jsonify({'error': 'Invalid input'}), 400

    X      = pd.DataFrame([row], columns=FEATURES)
    pred   = int(model.predict(X)[0])
    proba  = model.predict_proba(X)[0].tolist()

    return jsonify({
        'prediction': pred,
        'label':      LABELS[pred],
        'confidence': round(max(proba), 4),
        'probabilities': {LABELS[i]: round(p, 4) for i, p in enumerate(proba)},
    })


if __name__ == '__main__':
    app.run(debug=True, port=5003)
