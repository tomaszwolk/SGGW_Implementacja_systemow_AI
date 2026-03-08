"""Rozwiązanie — Etap 4: Ewaluacja modelu."""

import json
import pickle

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def main():
    print("Wczytywanie modelu...")
    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)

    print("Wczytywanie danych testowych...")
    test_df = pd.read_csv("data/test.csv")
    X_test = test_df.drop(columns=["survived"])
    y_test = test_df["survived"]

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    metrics = {"accuracy": round(accuracy, 4), "f1_score": round(f1, 4)}

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Metryki: accuracy={accuracy:.4f}, f1_score={f1:.4f}")
    print("Zapisano metrics.json")


if __name__ == "__main__":
    main()
