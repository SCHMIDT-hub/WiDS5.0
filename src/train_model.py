from load_data import load_upi_fraud_data
from preprocess import preprocess_data
from train_model import train_model


if __name__ == "__main__":
    # Load dataset
    df = load_upi_fraud_data()

    # Preprocess dataset
    preprocessor, X_train, X_test, y_train, y_test = preprocess_data(df)

    # Apply preprocessing
    X_train_processed = preprocessor.fit_transform(X_train)

    # Train model
    model = train_model(X_train_processed, y_train)

    # Simple sanity checks
    print("Model trained successfully!")
    print("Model type:", type(model))
    print("Number of training samples:", X_train_processed.shape[0])
    print("Number of features after preprocessing:", X_train_processed.shape[1])



