"""Trening i zapis dwóch modeli (RandomForest + LogisticRegression) do BentoML."""

import bentoml
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def load_and_preprocess_titanic():
    data = fetch_openml(data_id=40945, as_frame=True)
    df = data.frame
    df = df.drop(columns=["name", "ticket", "cabin", "body", "boat", "home.dest"])
    df["age"] = df["age"].fillna(df["age"].median())
    df["fare"] = df["fare"].fillna(df["fare"].median())
    df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])
    df["sex"] = df["sex"].map({"male": 0, "female": 1}).astype(int)
    df = pd.get_dummies(df, columns=["embarked"], drop_first=True)
    X = df.drop(columns=["survived"])
    y = df["survived"].astype(int)
    return X, y


def main():
    X, y = load_and_preprocess_titanic()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    signatures = {
        "predict": {"batchable": True, "batch_dim": 0},
        "predict_proba": {"batchable": True, "batch_dim": 0},
    }

    # Model 1: RandomForest
    rf = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)
    rf.fit(X_train, y_train)
    rf_accuracy = rf.score(X_test, y_test)

    rf_tag = bentoml.sklearn.save_model(
        "titanic_rf",
        rf,
        signatures=signatures,
        metadata={"accuracy": rf_accuracy, "model_type": "RandomForest"},
    )
    print(f"RandomForest zapisany: {rf_tag} (accuracy={rf_accuracy:.4f})")

    # Model 2: LogisticRegression
    lr = LogisticRegression(C=1.0, max_iter=200, random_state=42)
    lr.fit(X_train, y_train)
    lr_accuracy = lr.score(X_test, y_test)

    lr_tag = bentoml.sklearn.save_model(
        "titanic_logreg",
        lr,
        signatures=signatures,
        metadata={"accuracy": lr_accuracy, "model_type": "LogisticRegression"},
    )
    print(f"LogisticRegression zapisany: {lr_tag} (accuracy={lr_accuracy:.4f})")


if __name__ == "__main__":
    main()
