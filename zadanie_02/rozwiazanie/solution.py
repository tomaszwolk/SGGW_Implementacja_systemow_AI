"""
Rozwiązanie zadania 02 — MLflow Tracking: porównanie strategii imputacji.
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
    """Pobiera Titanic z OpenML i wykonuje preprocessing."""
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

    # 2. Trening modelu
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

    # 4. Logowanie do MLflow
    mlflow.set_experiment("titanic-preprocessing-comparison")

    with mlflow.start_run(run_name=f"rf-imputation-{fill_strategy}"):
        # Parametry
        mlflow.log_params(params)
        mlflow.log_param("imputation_strategy", fill_strategy)

        # Metryki
        mlflow.log_metrics(metrics)

        # Model z sygnaturą
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
        )

        # Artefakty
        tmpdir = tempfile.mkdtemp()

        # Macierz pomyłek
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=["Nie przeżył", "Przeżył"]
        )
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        disp.plot(ax=ax_cm, cmap="Blues")
        ax_cm.set_title(f"Macierz pomyłek — imputacja: {fill_strategy}")
        cm_path = os.path.join(tmpdir, "confusion_matrix.png")
        fig_cm.savefig(cm_path, bbox_inches="tight", dpi=150)
        plt.close(fig_cm)
        mlflow.log_artifact(cm_path, artifact_path="plots")

        # Krzywa ROC
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
        ax_roc.plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {roc_auc:.3f}")
        ax_roc.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title(f"Krzywa ROC — imputacja: {fill_strategy}")
        ax_roc.legend(loc="lower right")
        roc_path = os.path.join(tmpdir, "roc_curve.png")
        fig_roc.savefig(roc_path, bbox_inches="tight", dpi=150)
        plt.close(fig_roc)
        mlflow.log_artifact(roc_path, artifact_path="plots")

        # Tag
        mlflow.set_tag("imputation", fill_strategy)

        # Sprzątanie
        os.remove(cm_path)
        os.remove(roc_path)
        os.rmdir(tmpdir)

        print(f"  Run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    run_experiment("median")
    run_experiment("mean")
    print("\nGotowe! Otwórz MLflow UI: mlflow ui")
