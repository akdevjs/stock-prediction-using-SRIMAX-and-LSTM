import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pickle
import os

data = pd.read_csv('data-set.csv')
data['Date'] = pd.to_datetime(data['Date'])

data_per_product = data.pivot_table(index='Date', columns='Product Name', values='Quantity Used', fill_value=0)

model_dir = 'models/'
os.makedirs(model_dir, exist_ok=True)

for product in data_per_product.columns:
    product_name = product.lower()  # Ensure product name is in lowercase
    product_dir = f'{model_dir}/{product_name}'
    os.makedirs(product_dir, exist_ok=True)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_per_product[[product]])

    # LSTM Model Training
    look_back = 1
    X, Y = [], []
    for i in range(len(scaled_data) - look_back):
        X.append(scaled_data[i:(i + look_back), 0])
        Y.append(scaled_data[i + look_back, 0])
    X = np.array(X).reshape(-1, 1, look_back)
    Y = np.array(Y)

    model = Sequential([LSTM(4, input_shape=(1, look_back)), Dense(1)])
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, Y, epochs=100, batch_size=1, verbose=0)
    model.save(f'{product_dir}/lstm_model.h5')
    pickle.dump(scaler, open(f'{product_dir}/scaler.pkl', 'wb'))

    # SARIMAX Model Training
    sarimax_model = SARIMAX(data_per_product[[product]], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit(disp=0)
    pickle.dump(sarimax_model, open(f'{product_dir}/sarimax_model.pkl', 'wb'))
