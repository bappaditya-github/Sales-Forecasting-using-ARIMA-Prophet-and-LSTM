# `src/` Directory

This directory contains the core source code for the Sales Forecasting project.

## Structure

```plaintext
src/
├── data_loader.py       # Code to download or generate datasets
├── data_processor.py    # Code to clean and preprocess the datasets
├── model_trainer.py     # Code to train forecasting models like ARIMA, Prophet, LSTM
├── model_evaluator.py   # Code to evaluate the performance of models
├── utils.py             # Utility functions for the project
```

## Usage
- Each script is modularized for clarity and can be imported as needed.
- Ensure the data files are placed in the `data/` directory before running the scripts.
