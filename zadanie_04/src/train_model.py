"""Etap 3: Trening modelu RandomForest (bez metryk — metryki liczy evaluate)."""

import os
import pickle

import yaml
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def main():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    model_params = params["model"]

    print("Wczytywanie danych treningowych...")
    train_df = pd.read_csv("data/train.csv")
    X_train = train_df.drop(columns=["survived"])
    y_train = train_df["survived"]

    print(f"Trening modelu: n_estimators={model_params['n_estimators']}, max_depth={model_params['max_depth']}")
    model = RandomForestClassifier(
        n_estimators=model_params["n_estimators"],
        max_depth=model_params["max_depth"],
        random_state=model_params["random_state"],
    )
    model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Model zapisany do models/model.pkl")


if __name__ == "__main__":
    main()
