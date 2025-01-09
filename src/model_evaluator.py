import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMAResults
from prophet import Prophet
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
import os
import pickle
import warnings
import tqdm

processed_data_path = "../data/processed/processed_sales_data.csv"
model_path = "models"

def load_processed_data():
    return pd.read_csv(processed_data_path, parse_dates=["Date"])

def evaluate_arima(data):
    model = ARIMAResults.load(f"{model_path}/arima_model.pkl")
    predictions = model.forecast(steps=len(data))
    mse = mean_squared_error(data['Sales'], predictions)
    print(f"ARIMA Model MSE: {mse}")

def evaluate_prophet(data):
    data.rename(columns={"Date": "ds", "Sales": "y"}, inplace=True)
    with open(f"{model_path}/prophet_model.pkl", "rb") as f:
        model = pickle.load(f)
    future = model.make_future_dataframe(periods=len(data))
    forecast = model.predict(future)
    mse = mean_squared_error(data['y'], forecast['yhat'][:len(data)])
    print(f"Prophet Model MSE: {mse}")

def evaluate_lstm(data):
    model = load_model(f"{model_path}/lstm_model.keras")
    data_values = data['Sales'].values.reshape(-1, 1)
    predictions = model.predict(data_values[:-1])
    mse = mean_squared_error(data_values[1:], predictions)
    print(f"LSTM Model MSE: {mse}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=tqdm.TqdmWarning)
    data = load_processed_data()
    prophet_data = load_processed_data()
    evaluate_arima(data.copy())
    evaluate_prophet(prophet_data)
    evaluate_lstm(data.copy())
