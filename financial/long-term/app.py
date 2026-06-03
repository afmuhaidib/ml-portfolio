"""
Flask app — Long-Term Stock Signal
Route: GET /  → HTML page
Route: POST /analyze  → JSON { ticker, current_price, predicted_price, fair_value, signal, upside_pct }
"""

import sys
from pathlib import Path
from flask import Flask, render_template, request, jsonify

sys.path.insert(0, str(Path(__file__).parent))
from predict import predict

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    data   = request.get_json(silent=True) or {}
    ticker = (data.get('ticker') or '').strip().upper()
    if not ticker:
        return jsonify({'error': 'Ticker is required'}), 400
    try:
        result = predict(ticker)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5001)
