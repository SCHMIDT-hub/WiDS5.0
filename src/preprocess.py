from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def preprocess_data(df):
    # Separate features and target
    X = df.drop(columns=["fraud_risk", "Id", "upi_number"])
    y = df["fraud_risk"]

    # Define feature types
    numerical_features = [
        "trans_hour",
        "trans_day",
        "trans_month",
        "trans_year",
        "age",
        "trans_amount",
    ]

    categorical_features = [
        "category",
        "state",
        "zip",
    ]

    # Preprocessing pipelines
    numeric_transformer = Pipeline(
        steps=[("scaler", StandardScaler())]
    )

    categorical_transformer = Pipeline(
        steps=[("encoder", OneHotEncoder(handle_unknown="ignore"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return preprocessor, X_train, X_test, y_train, y_test
