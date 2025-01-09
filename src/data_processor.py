import pandas as pd
import os

data_path = "../data/raw"
processed_data_path = "../data/processed"
os.makedirs(processed_data_path, exist_ok=True)

def load_data(filename):
    file_path = os.path.join(data_path, filename)
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values(by='Date')
    data = data.dropna()
    return data

def save_processed_data(data, filename="processed_sales_data.csv"):
    file_path = os.path.join(processed_data_path, filename)
    data.to_csv(file_path, index=False)
    print(f"Processed data saved to {file_path}")

if __name__ == "__main__":
    data = load_data("synthetic_sales_data.csv")
    processed_data = preprocess_data(data)
    save_processed_data(processed_data)
