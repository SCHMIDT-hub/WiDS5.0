from load_data import load_upi_fraud_data
from preprocess import preprocess_data
from train_model import train_model
from evaluate import evaluate_model


if __name__ == "__main__":
    # Load dataset
    df = load_upi_fraud_data()

    # Preprocess
    preprocessor, X_train, X_test, y_train, y_test = preprocess_data(df)
    X_train_p = preprocessor.fit_transform(X_train)
    X_test_p = preprocessor.transform(X_test)

    # Train model
    model = train_model(X_train_p, y_train)

    # Evaluate model
    metrics = evaluate_model(model, X_test_p, y_test)

    print("Evaluation test successful!")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
