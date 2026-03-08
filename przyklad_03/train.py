"""
Skrypt treningowy dla klasyfikatora Random Forest na zbiorze Titanic.
Wykorzystuje MLflow do logowania parametrow, metryk i modelu.
Uruchamiany jako entry point projektu MLflow Projects.
"""

import argparse

import mlflow
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


def parse_args():
    """Parsowanie argumentow wiersza polecen."""
    parser = argparse.ArgumentParser(
        description="Trening klasyfikatora Random Forest na zbiorze Titanic"
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Liczba drzew w lesie losowym (domyslnie: 100)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=5,
        help="Maksymalna glebokosc drzewa (domyslnie: 5)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Ziarno losowosci dla reprodukowalnosci (domyslnie: 42)",
    )
    return parser.parse_args()


def load_and_preprocess_data():
    """
    Wczytanie zbioru Titanic z OpenML i przygotowanie danych.
    Usuwa kolumny nieistotne, uzupelnia braki, koduje zmienne kategoryczne.
    """
    # Pobranie zbioru danych Titanic z OpenML (id=40945)
    data = fetch_openml(data_id=40945, as_frame=True)
    df = data.frame

    # Usuniecie kolumn, ktore nie sa przydatne do modelowania
    df = df.drop(columns=["name", "ticket", "cabin", "body", "boat", "home.dest"])

    # Uzupelnienie brakujacych wartosci medianą lub dominanta
    df["age"] = df["age"].fillna(df["age"].median())
    df["fare"] = df["fare"].fillna(df["fare"].median())
    df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

    # Kodowanie zmiennej plci na wartosci numeryczne
    df["sex"] = df["sex"].map({"male": 0, "female": 1}).astype(int)

    # One-hot encoding dla portu zaokretowania
    df = pd.get_dummies(df, columns=["embarked"], drop_first=True)

    # Podzial na cechy (X) i zmienna docelowa (y)
    X = df.drop(columns=["survived"])
    y = df["survived"].astype(int)

    return X, y


def main():
    """Glowna funkcja treningowa z logowaniem do MLflow."""
    args = parse_args()

    # Wczytanie i przetworzenie danych
    X, y = load_and_preprocess_data()

    # Podzial na zbior treningowy i testowy (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.random_state
    )

    # Rozpoczecie runu MLflow
    with mlflow.start_run():
        # Logowanie hiperparametrow
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("random_state", args.random_state)

        # Utworzenie i trening modelu Random Forest
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=args.random_state,
        )
        model.fit(X_train, y_train)

        # Predykcja na zbiorze testowym
        y_pred = model.predict(X_test)

        # Obliczenie metryk jakosci modelu
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Logowanie metryk do MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)

        print(f"Dokladnosc (accuracy): {accuracy:.4f}")
        print(f"Miara F1:              {f1:.4f}")

        # Utworzenie sygnatury modelu na podstawie danych wejsciowych i wyjsciowych
        signature = mlflow.models.infer_signature(X_train, model.predict(X_train))

        # Logowanie modelu z sygnatura do MLflow
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
        )

        print("Model zapisany w MLflow.")


if __name__ == "__main__":
    main()
