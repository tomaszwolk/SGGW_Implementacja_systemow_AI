"""
Zadanie 05 — Optuna + MLflow: optymalizacja hiperparametrów.

Uzupełnij miejsca oznaczone TODO.
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
    # 1. Przygotowanie danych
    X, y = load_and_preprocess_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 2. Funkcja celu
    def objective(trial):
        # TODO 1: Zdefiniuj hiperparametry za pomocą trial.suggest_*
        # n_estimators = trial.suggest_int("n_estimators", 50, 300, step=50)
        # max_depth = trial.suggest_int("max_depth", 3, 15)
        # min_samples_split = trial.suggest_int("min_samples_split", 2, 10)

        # TODO 2: Stwórz RandomForestClassifier z parametrami z trial
        # model = RandomForestClassifier(...)

        # TODO 3: Ewaluuj z cross_val_score (cv=5, scoring="accuracy")
        # scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
        # return scores.mean()

        pass

    # 3. Konfiguracja MLflow
    mlflow.set_experiment("titanic-optuna")

    # TODO 4: Stwórz MLflowCallback
    # mlflow_callback = MLflowCallback(
    #     tracking_uri=mlflow.get_tracking_uri(),
    #     metric_name="cv_accuracy",
    # )

    # TODO 5: Stwórz study i uruchom optymalizację (15 prób)
    # study = optuna.create_study(direction="maximize")
    # study.optimize(objective, n_trials=15, callbacks=[mlflow_callback])

    # TODO 6: Wytrenuj najlepszy model i zaloguj do MLflow
    # best_model = RandomForestClassifier(**study.best_params, random_state=42)
    # best_model.fit(X_train, y_train)
    # with mlflow.start_run(run_name="best-model"):
    #     mlflow.log_params(study.best_params)
    #     mlflow.log_metric("cv_accuracy", study.best_value)
    #     mlflow.log_metric("test_accuracy", accuracy_score(y_test, best_model.predict(X_test)))
    #     signature = infer_signature(X_train, best_model.predict(X_train))
    #     mlflow.sklearn.log_model(best_model, "model", signature=signature)

    pass


if __name__ == "__main__":
    main()
