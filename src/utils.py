import os
import pandas as pd

def ensure_directory_exists(directory):
    """Ensure the directory exists, create if it doesn't."""
    os.makedirs(directory, exist_ok=True)

def load_csv(filepath):
    """Load a CSV file into a Pandas DataFrame."""
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None

def save_csv(data, filepath):
    """Save a Pandas DataFrame to a CSV file."""
    ensure_directory_exists(os.path.dirname(filepath))
    data.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")
