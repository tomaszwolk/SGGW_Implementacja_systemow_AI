"""
Zadanie 03 — MLflow Projects: skrypt treningowy do opakowania.

TODO: Dodaj argparse, aby parametry (n_estimators, max_depth, imputation)
      były przekazywane z linii poleceń zamiast hardkodowane.
"""

import os
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    # TODO: Dodaj argparse z argumentami:
    #   --n-estimators (int, default=100)
    #   --max-depth (int, default=5)
    #   --imputation (str, default="median")
    # Użyj: import argparse; parser = argparse.ArgumentParser(); ...

    # --- Na razie hardkodowane wartości (zamień na args z argparse) ---
    n_estimators = 100
    max_depth = 5
    imputation = "median"

    X, y = load_and_preprocess_data(imputation)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "random_state": 42,
        "imputation": imputation,
    }

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Gdy uruchamiamy przez `mlflow run`, MLflow sam tworzy run —
    # nie wolno wtedy zmieniać experiment, bo będzie konflikt.
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

        print(f"Parametry: n_estimators={n_estimators}, max_depth={max_depth}, imputation={imputation}")
        print(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        print(f"Run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
