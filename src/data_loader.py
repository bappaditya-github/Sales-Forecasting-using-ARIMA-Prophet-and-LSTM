import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

data_path = "../data/raw"
os.makedirs(data_path, exist_ok=True)

def generate_sales_data(start_date="2024-01-01", days=365):
    np.random.seed(42)
    dates = [datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=i) for i in range(days)]
    sales = np.random.poisson(lam=100, size=days) + np.sin(np.linspace(0, 50, days)) * 20
    data = pd.DataFrame({"Date": dates, "Sales": sales})
    data.to_csv(f"{data_path}/synthetic_sales_data.csv", index=False)
    print(f"Synthetic sales data generated and saved to {data_path}")

if __name__ == "__main__":
    generate_sales_data()
