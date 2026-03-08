"""
Zadanie 02 — MLflow Tracking: porównanie strategii imputacji.

Uzupełnij miejsca oznaczone TODO, aby zalogować eksperymenty do MLflow.
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
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split


def load_and_preprocess_data(fill_strategy="median"):
    """Pobiera Titanic z OpenML i wykonuje preprocessing.

    Args:
        fill_strategy: "median" lub "mean" — strategia uzupełniania braków w kolumnie age.
    """
    data = fetch_openml(data_id=40945, as_frame=True)
    df = data.frame

    df = df.drop(columns=["name", "ticket", "cabin", "body", "boat", "home.dest"])

    if fill_strategy == "median":
        df["age"] = df["age"].fillna(df["age"].median())
    elif fill_strategy == "mean":
        df["age"] = df["age"].fillna(df["age"].mean())
    else:
        raise ValueError(f"Nieznana strategia: {fill_strategy}")

    df["fare"] = df["fare"].fillna(df["fare"].median())
    df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

    df["sex"] = df["sex"].map({"male": 0, "female": 1}).astype(int)
    df = pd.get_dummies(df, columns=["embarked"], drop_first=True)

    X = df.drop(columns=["survived"])
    y = df["survived"].astype(int)

    return X, y


def run_experiment(fill_strategy):
    """Trenuje model z daną strategią imputacji i loguje do MLflow."""

    print(f"\n{'='*60}")
    print(f"Strategia imputacji: {fill_strategy}")
    print(f"{'='*60}")

    # 1. Przygotowanie danych
    X, y = load_and_preprocess_data(fill_strategy)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 2. Trening modelu (te same hiperparametry dla obu wariantów)
    params = {
        "n_estimators": 100,
        "max_depth": 7,
        "random_state": 42,
    }
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    # 3. Predykcja i metryki
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
    }

    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    # ── TODO 1: Ustaw eksperyment MLflow ──────────────────────────────────
    # Użyj mlflow.set_experiment() z nazwą "titanic-preprocessing-comparison"

    # ── TODO 2: Rozpocznij run z nazwą zawierającą strategię ──────────────
    # Użyj: with mlflow.start_run(run_name=f"rf-imputation-{fill_strategy}"):

        # ── TODO 3: Zaloguj parametry (hiperparametry + strategia) ────────
        # Zaloguj params oraz dodatkowy parametr "imputation_strategy"

        # ── TODO 4: Zaloguj metryki ──────────────────────────────────────

        # ── TODO 5: Zaloguj model z sygnaturą ────────────────────────────
        # Użyj infer_signature() i mlflow.sklearn.log_model()

        # ── TODO 6: Zaloguj macierz pomyłek jako artefakt PNG ────────────
        # Użyj ConfusionMatrixDisplay, zapisz do pliku tymczasowego,
        # zaloguj przez mlflow.log_artifact(), usuń plik

        # ── TODO 7: Zaloguj krzywą ROC jako artefakt PNG ─────────────────
        # Użyj roc_curve() i auc(), narysuj wykres matplotlib,
        # zapisz do pliku, zaloguj, usuń

        # ── TODO 8: Ustaw tag z nazwą strategii imputacji ────────────────
        # mlflow.set_tag("imputation", ...)

        # print(f"  Run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    run_experiment("median")
    run_experiment("mean")
    print("\nGotowe! Otwórz MLflow UI: mlflow ui")
