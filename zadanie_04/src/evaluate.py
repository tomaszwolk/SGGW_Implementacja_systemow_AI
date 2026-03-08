"""
Etap 4: Ewaluacja modelu na zbiorze testowym.

Uzupełnij miejsca oznaczone TODO.
"""

import json
import pickle

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def main():
    # TODO 1: Wczytaj model z pliku models/model.pkl
    # Użyj: with open("models/model.pkl", "rb") as f: model = pickle.load(f)

    # TODO 2: Wczytaj dane testowe z data/test.csv
    # Rozdziel na X_test (cechy) i y_test (kolumna "survived")

    # TODO 3: Wykonaj predykcję
    # y_pred = model.predict(X_test)

    # TODO 4: Oblicz metryki
    # accuracy = accuracy_score(y_test, y_pred)
    # f1 = f1_score(y_test, y_pred)

    # TODO 5: Zapisz metryki do metrics.json
    # metrics = {"accuracy": round(accuracy, 4), "f1_score": round(f1, 4)}
    # with open("metrics.json", "w") as f: json.dump(metrics, f, indent=2)

    pass


if __name__ == "__main__":
    main()
