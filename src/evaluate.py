from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model on test data.

    Parameters
    ----------
    model : trained sklearn model
    X_test : array-like
        Preprocessed test features
    y_test : array-like
        True labels (fraud_risk)

    Returns
    -------
    dict
        accuracy, f1_score, roc_auc
    """
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    return metrics


def cross_validate_model(model, X, y, cv=5):
    """
    Perform cross-validation using ROC-AUC.

    Parameters
    ----------
    model : sklearn model
    X : array-like
        Full feature matrix
    y : array-like
        Target vector (fraud_risk)
    cv : int
        Number of folds

    Returns
    -------
    float
        Mean ROC-AUC score across folds
    """
    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="roc_auc"
    )
    return scores.mean()
