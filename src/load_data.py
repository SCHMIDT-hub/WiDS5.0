import pandas as pd
from pathlib import Path


def load_upi_fraud_data():
    """
    Loads the UPI fraud dataset from the data/raw directory.

    Returns:
        pd.DataFrame: Loaded dataset
    """
    data_path = Path("data/raw/upi_fraud_dataset.csv")

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. "
            "Please ensure the CSV is placed in data/raw/"
        )

    df = pd.read_csv(data_path)
    return df


if __name__ == "__main__":
    df = load_upi_fraud_data()
    print("Dataset loaded successfully")
    print(df.head())
