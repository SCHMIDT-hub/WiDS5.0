from src.load_data import load_upi_fraud_data
from src.preprocess import preprocess_data
from src.train_model import train_model
from src.evaluate import evaluate_model, cross_validate_model


def main():
    print("Starting UPI Fraud Detection Pipeline...\n")

    # Load dataset
    df = load_upi_fraud_data()

    # Preprocess
    preprocessor, X_train, X_test, y_train, y_test = preprocess_data(df)
    X_train_p = preprocessor.fit_transform(X_train)
    X_test_p = preprocessor.transform(X_test)

    # Train model
    model = train_model(X_train_p, y_train)

    # Evaluate on test set
    metrics = evaluate_model(model, X_test_p, y_test)

    print("MODEL EVALUATION RESULTS")
    print("-" * 30)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # Cross-validation
    X = df.drop(columns=["fraud_risk", "Id", "upi_number"])
    y = df["fraud_risk"]
    X_full = preprocessor.fit_transform(X)

    cv_score = cross_validate_model(model, X_full, y)
    print(f"Cross-validated ROC-AUC: {cv_score:.4f}")

    # Save results
    with open("results.txt", "w") as f:
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
        f.write(f"cv_roc_auc: {cv_score:.4f}\n")

    print("\nPipeline completed successfully!")
    print("Results saved to results.txt")


if __name__ == "__main__":
    main()
