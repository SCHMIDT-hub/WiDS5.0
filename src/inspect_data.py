from load_data import load_upi_fraud_data


def inspect_dataset(df):
    print("Shape of dataset:", df.shape)
    print("\nColumns:\n", df.columns.tolist())
    print("\nData types:\n", df.dtypes)
    print("\nMissing values:\n", df.isnull().sum())


if __name__ == "__main__":
    df = load_upi_fraud_data()
    inspect_dataset(df)
