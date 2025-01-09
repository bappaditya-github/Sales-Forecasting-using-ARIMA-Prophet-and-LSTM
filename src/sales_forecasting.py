import pandas as pd
from statsmodels.tsa.arima.model import ARIMAResults
from prophet import Prophet
from tensorflow.keras.models import load_model
import numpy as np
import os

processed_data_path = "../data/processed/processed_sales_data.csv"
model_path = "models"
forecast_results_path = "results"
os.makedirs(forecast_results_path, exist_ok=True)

def load_processed_data():
    return pd.read_csv(processed_data_path, parse_dates=["Date"])

def forecast_arima(steps=30):
    data = load_processed_data()
    model = ARIMAResults.load(f"{model_path}/arima_model.pkl")
    forecast = model.forecast(steps=steps)
    forecast_df = pd.DataFrame({"Date": pd.date_range(data["Date"].max(), periods=steps + 1)[1:], "Forecasted_Sales": forecast})
    forecast_df.to_csv(f"{forecast_results_path}/arima_forecast.csv", index=False)
    print("ARIMA forecast saved.")

def forecast_prophet(steps=30):
    data = load_processed_data()
    data.rename(columns={"Date": "ds", "Sales": "y"}, inplace=True)
    model = Prophet()
    model.fit(data)
    future = model.make_future_dataframe(periods=steps)
    forecast = model.predict(future)
    forecast[['ds', 'yhat']].tail(steps).to_csv(f"{forecast_results_path}/prophet_forecast.csv", index=False)
    print("Prophet forecast saved.")

def forecast_lstm(steps=30):
    data = load_processed_data()
    model = load_model(f"{model_path}/lstm_model.keras")
    data_values = data['Sales'].values.reshape(-1, 1)
    predictions = []
    for _ in range(steps):
        prediction = model.predict(data_values[-1].reshape(1, 1, 1))
        predictions.append(prediction[0, 0])
        data_values = np.append(data_values, prediction[0, 0]).reshape(-1, 1)
    forecast_df = pd.DataFrame({"Date": pd.date_range(data["Date"].max(), periods=steps + 1)[1:], "Forecasted_Sales": predictions})
    forecast_df.to_csv(f"{forecast_results_path}/lstm_forecast.csv", index=False)
    print("LSTM forecast saved.")

if __name__ == "__main__":
    forecast_arima()
    forecast_prophet()
    forecast_lstm()
