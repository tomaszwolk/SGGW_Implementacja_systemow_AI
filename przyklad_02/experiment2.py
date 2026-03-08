import os
import tempfile
import shutil
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.data
import pandas as pd
from mlflow.models import infer_signature
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


def load_and_preprocess_data():
    """Pobiera zbiór Titanic z OpenML i wykonuje preprocessing."""

    # Pobranie danych z OpenML (id=40945 — Titanic)
    data = fetch_openml(data_id=40945, as_frame=True)
    df = data.frame

    # Usunięcie kolumn, które nie wnoszą wartości predykcyjnej
    df = df.drop(columns=["name", "ticket", "cabin", "body", "boat", "home.dest"])

    # Uzupełnienie braków wartością mediany / dominanty
    df["age"] = df["age"].fillna(df["age"].median())
    df["fare"] = df["fare"].fillna(df["fare"].median())
    df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

    # Zamiana zmiennych kategorycznych na numeryczne
    df["sex"] = df["sex"].map({"male": 0, "female": 1}).astype(int)
    df = pd.get_dummies(df, columns=["embarked"], drop_first=True)

    # Podział na cechy (X) i zmienną docelową (y)
    X = df.drop(columns=["survived"])
    y = df["survived"].astype(int)

    return X, y

def main():
    X, y = load_and_preprocess_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    train_dataset = mlflow.data.from_pandas(train_data, name="titanic-train", targets="survived")
    test_dataset = mlflow.data.from_pandas(test_data, name="titanic-test", targets="survived")

    mlflow.set_experiment("titanic-classification")

    with mlflow.start_run(run_name="logistic-regression"):
        mlflow.log_input(train_dataset, context="train")
        mlflow.log_input(test_dataset, context="test")

        params = {
            "C": 1.0,
            "penalty": "l2",
        }
        mlflow.log_params(params)
        model = LogisticRegression(**params)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
        }

        mlflow.log_metrics(metrics)

        print(metrics)

        tmpdir = tempfile.mkdtemp()
        try:
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Nie przeżył", "Przeżył"])
            fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
            disp.plot(ax=ax_cm, cmap="Blues")
            ax_cm.set_title("Macierz pomyłek — Random Forest")
            cm_path = os.path.join(tmpdir, "confusion_matrix.png")
            fig_cm.savefig(cm_path, bbox_inches="tight", dpi=150)
            plt.close(fig_cm)
            mlflow.log_artifact(cm_path, artifact_path="plots")
        finally:
            shutil.rmtree(tmpdir)

        mlflow.log_artifact(__file__, artifact_path="code")

        signature = infer_signature(X_test, y_pred)
        mlflow.sklearn.log_model(model, "model", signature=signature)
if __name__ == "__main__":
    main()