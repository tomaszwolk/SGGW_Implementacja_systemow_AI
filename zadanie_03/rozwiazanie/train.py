"""
Rozwiązanie zadania 03 — MLflow Projects: skrypt treningowy z argparse.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


def load_and_preprocess_data(fill_strategy="median"):
    """Pobiera Titanic z OpenML i wykonuje preprocessing."""
    data = fetch_openml(data_id=40945, as_frame=True)
    df = data.frame

    df = df.drop(columns=["name", "ticket", "cabin", "body", "boat", "home.dest"])

    if fill_strategy == "median":
        df["age"] = df["age"].fillna(df["age"].median())
    else:
        df["age"] = df["age"].fillna(df["age"].mean())

    df["fare"] = df["fare"].fillna(df["fare"].median())
    df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])
    df["sex"] = df["sex"].map({"male": 0, "female": 1}).astype(int)
    df = pd.get_dummies(df, columns=["embarked"], drop_first=True)

    X = df.drop(columns=["survived"])
    y = df["survived"].astype(int)
    return X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--imputation", type=str, default="median")
    args = parser.parse_args()

    X, y = load_and_preprocess_data(args.imputation)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    params = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "random_state": 42,
        "imputation": args.imputation,
    }

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Gdy uruchamiamy przez `mlflow run`, MLflow sam tworzy run i ustawia
    # MLFLOW_RUN_ID w środowisku — nie wolno wtedy wywoływać set_experiment(),
    # bo zmieni experiment i start_run() zgłosi konflikt.
    if "MLFLOW_RUN_ID" not in os.environ:
        mlflow.set_experiment("titanic-mlflow-project")

    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)

        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
        )

        print(f"Parametry: n_estimators={args.n_estimators}, max_depth={args.max_depth}, imputation={args.imputation}")
        print(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        print(f"Run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
