# Sales-Forecasting-using-ARIMA-Prophet-and-LSTM

# Sales Forecasting using ARIMA, Prophet, and LSTM

## Project Overview
This project focuses on time series forecasting for sales data using ARIMA, Prophet, and LSTM models. It includes exploratory data analysis (EDA), feature engineering, and model comparison.

## Project Structure
```
├── data/                   # Raw and processed datasets
├── notebooks/              # Jupyter notebooks for EDA and model building
├── src/                    # Python scripts for data preprocessing and model training
├── results/                # Model results and performance metrics
├── README.md               # Project overview and instructions
├── requirements.txt        # Required libraries for the project
└── .gitignore              # Files to be ignored in the repository
```

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone <repository_link>
   cd sales_forecasting
   ```
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```
3. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```

## Data
- The data should be placed in the `data/` directory.
- Ensure the dataset includes time series data with a timestamp and sales column.

## Steps
### 1. Exploratory Data Analysis (EDA)
- Data visualization and summary statistics
- Trend, seasonality, and stationarity checks

### 2. Feature Engineering
- Creating lag features, rolling statistics, and differencing for stationarity

### 3. Model Training
- **ARIMA:** Statistical model for time series forecasting
- **Prophet:** Trend and seasonality decomposition
- **LSTM:** Neural network model for sequential data

### 4. Model Evaluation
- Metrics: RMSE, MAE, MAPE
- Visualization of predictions vs. actual values

## Results
- The results will be stored in the `results/` directory with performance metrics and plots.

## Contributions
- Fork the repository and create a pull request for contributions.

## License
- MIT License

## Contact
- For inquiries, contact bappadityaghosh.tn@gmail.com.
