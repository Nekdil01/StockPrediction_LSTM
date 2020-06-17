from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import json
import datetime
from datetime import timedelta

app = Flask(__name__)
tens_init = Sequential()
STEPS = 60


def processData(company):
    # Читаем файл
    ticker = yf.Ticker(company)

    df = ticker.history(period="10y", interval="1d")

    # Верх данных
    df.head()

    labels = []
    for i in df.index:
        labels.append(datetime.date(i.year, i.month, i.day).strftime("%Y-%m-%d"))
    

    data = df.sort_index(ascending=True, axis=0)

    data.drop('Open', axis=1, inplace=True)
    data.drop('High', axis=1, inplace=True)
    data.drop('Low', axis=1, inplace=True)
    data.drop('Volume', axis=1, inplace=True)
    data.drop('Dividends', axis=1, inplace=True)
    data.drop('Stock Splits', axis=1, inplace=True)

    new_data = data

    dataset = new_data.values

    train = dataset[0:, :]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    x_train, y_train = [], []
    for i in range(60, len(train)):
        x_train.append(scaled_data[i - 60:i, 0])
        y_train.append(scaled_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(60, 1)))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=50, batch_size=60)

    inputs = new_data[len(new_data) - 60:].values
    inputs = inputs.reshape(-1, 1)
    inputs = scaler.transform(inputs)
    last_date = df.index[len(df) - 1] + timedelta(days=1)
    last_date = datetime.date(last_date.year, last_date.month, last_date.day)
    predictions = pd.DataFrame(index=range(0, 100), columns=['Date', 'Close'])

    for i in range(0, 100):
        #Form input for model, which contains last 60 days including prices predicte with model
        pred_input = inputs[len(inputs) - 60:len(inputs), 0]
        pred_input = np.array(pred_input)
        pred_input = np.reshape(pred_input, (1, 60, 1))
        #Predicting 61st day
        closing_price = model.predict(pred_input)
        inputs = np.append(inputs, closing_price)
        inputs = np.reshape(inputs, (inputs.shape[0], 1))
        closing_price = scaler.inverse_transform(closing_price)
        predictions['Date'][i] = last_date + timedelta(days=i)
        predictions['Close'][i] = closing_price[0, 0]
    predictions.index = predictions['Date']
    predictions = predictions.drop('Date', axis=1)

    real_tograph = pd.DataFrame(index=range(0, len(data)), columns=['x', 'y'])
    for i in range(0, len(new_data)):
        real_tograph['x'][i] = labels[i]
        real_tograph['y'][i] = new_data['Close'][i]
    data_raw = real_tograph.to_dict(orient='records')

    for i in predictions.index:
        labels.append(datetime.date(i.year, i.month, i.day).strftime("%Y-%m-%d"))

    pred_tograph = pd.DataFrame(index=range(0, len(predictions)), columns=['x', 'y'])
    for i in range(0, len(predictions)):
        pred_tograph['x'][i] = predictions.index[i].strftime("%Y-%m-%d")
        pred_tograph['y'][i] = np.float64(predictions['Close'][i])
    data_predicted = pred_tograph.to_dict(orient='records')

    data_test = [labels, data_predicted, data_raw]
    return data_test


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/_get_labels/', methods=['POST'])
def _get_labels():
    return jsonify(processData(request.form['name']))


if __name__ == "__main__":

    import os
    app.run(port=int(os.environ.get("PORT", 5000)), host='0.0.0.0')
