"""
Rozwiązanie zadania 05 — Optuna + MLflow: optymalizacja hiperparametrów.
"""

import mlflow
import mlflow.sklearn
import optuna
import pandas as pd
from mlflow.models import infer_signature
from optuna.integration.mlflow import MLflowCallback
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score, train_test_split


def load_and_preprocess_data():
    """Pobiera Titanic z OpenML i wykonuje preprocessing."""
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
    X, y = load_and_preprocess_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 50, 300, step=50)
        max_depth = trial.suggest_int("max_depth", 3, 15)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 4)

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
        )

        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
        return scores.mean()

    mlflow.set_experiment("titanic-optuna")

    mlflow_callback = MLflowCallback(
        tracking_uri=mlflow.get_tracking_uri(),
        metric_name="cv_accuracy",
    )

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=15, callbacks=[mlflow_callback])

    print(f"\nNajlepsze parametry: {study.best_params}")
    print(f"Najlepsza CV accuracy: {study.best_value:.4f}")

    best_model = RandomForestClassifier(**study.best_params, random_state=42)
    best_model.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)

    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test F1: {test_f1:.4f}")

    with mlflow.start_run(run_name="best-model"):
        mlflow.log_params(study.best_params)
        mlflow.log_metric("cv_accuracy", study.best_value)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_f1", test_f1)

        signature = infer_signature(X_train, best_model.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            signature=signature,
        )

        print(f"Run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
