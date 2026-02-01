from sklearn.linear_model import LogisticRegression


def train_model(X_train, y_train):
    """
    Train a fraud detection model using Logistic Regression.

    Parameters
    ----------
    X_train : array-like
        Preprocessed training features
    y_train : array-like
        Training labels (fraud_risk)

    Returns
    -------
    model : trained sklearn model
    """
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)
    return model
