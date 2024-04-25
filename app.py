from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper
from sklearn.metrics import mean_squared_error
import pickle
import numpy as np
from datetime import datetime
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

model_dir = 'models/'
models = {}
scalers = {}
sarimax_models = {}

for product_dir in os.listdir(model_dir):
    product_name = product_dir  # Assume directory names are already in lowercase
    models[product_name] = load_model(f'{model_dir}{product_dir}/lstm_model.h5')
    scalers[product_name] = pickle.load(open(f'{model_dir}{product_dir}/scaler.pkl', 'rb'))
    sarimax_models[product_name] = pickle.load(open(f'{model_dir}{product_dir}/sarimax_model.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    content = request.json
    product_name = content['product_name'].lower()
    date = content['date']
    end_date = datetime.strptime(date, '%Y-%m-%d')

    if product_name not in sarimax_models:
        return jsonify({'error': 'Model for the specified product does not exist.'}), 404

    # SARIMAX Prediction and Accuracy
    sarimax_prediction = sarimax_models[product_name].get_prediction(start=end_date, end=end_date).predicted_mean.item()
    sarimax_error = np.sqrt(mean_squared_error([sarimax_models[product_name].fittedvalues[-1]], [sarimax_prediction]))
    sarimax_accuracy = max(0, 100 - (sarimax_error / max(1, abs(sarimax_prediction)) * 100))

    # LSTM Prediction and Accuracy
    scaler = scalers[product_name]
    last_known_value = scaler.transform([[0]])
    testX = np.reshape(last_known_value, (1, 1, 1))
    lstm_prediction = scaler.inverse_transform(models[product_name].predict(testX))[0][0]
    lstm_error = np.sqrt(mean_squared_error([sarimax_prediction], [lstm_prediction]))
    lstm_accuracy = max(0, 100 - (lstm_error / max(1, abs(lstm_prediction)) * 100))

    return jsonify({
        'product_name': product_name,
        'date': date,
        'sarimax_prediction': int(sarimax_prediction),
        'sarimax_accuracy': f'{sarimax_accuracy:.2f}%',
        'lstm_prediction': int(lstm_prediction),
        'lstm_accuracy': f'{lstm_accuracy:.2f}%'
    })

if __name__ == '__main__':
    app.run(debug=True)
