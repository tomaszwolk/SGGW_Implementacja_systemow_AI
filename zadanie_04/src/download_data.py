"""Etap 1: Pobieranie danych Titanic z OpenML."""

import os

import yaml
from sklearn.datasets import fetch_openml


def main():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    dataset_id = params["data"]["dataset_id"]
    print(f"Pobieranie danych z OpenML (id={dataset_id})...")

    data = fetch_openml(data_id=dataset_id, as_frame=True)
    df = data.frame

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/titanic.csv", index=False)
    print(f"Zapisano data/titanic.csv ({len(df)} wierszy)")


if __name__ == "__main__":
    main()
