import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense
import os
import pickle

processed_data_path = "../data/processed/processed_sales_data.csv"
model_path = "models"
os.makedirs(model_path, exist_ok=True)

def load_processed_data():
    return pd.read_csv(processed_data_path, parse_dates=["Date"])

def train_arima(data):
    model = ARIMA(data['Sales'], order=(5,1,0))
    model_fit = model.fit()
    model_fit.save(f"{model_path}/arima_model.pkl")
    print("ARIMA model trained and saved.")

def train_prophet(data):
    data.rename(columns={"Date": "ds", "Sales": "y"}, inplace=True)
    model = Prophet()
    model.fit(data)
    with open(f"{model_path}/prophet_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Prophet model trained and saved.")

def train_lstm(data):
    if 'Sales' in data.columns:
        data_values = data['Sales'].values.reshape(-1, 1)
        model = Sequential()
        model.add(Input(shape=(1, 1)))
        model.add(LSTM(50, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        model.fit(data_values[:-1], data_values[1:], epochs=10, batch_size=1)
        model.save(f"{model_path}/lstm_model.keras")
        print("LSTM model trained and saved.")
    else:
        print("Error: 'Sales' column not found in data.")

if __name__ == "__main__":
    data = load_processed_data()
    prophet_data = load_processed_data()
    train_arima(data)
    train_prophet(prophet_data)
    train_lstm(data)
