"""Etap 2: Preprocessing danych Titanic."""

import yaml
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    test_size = params["prepare"]["test_size"]
    random_state = params["prepare"]["random_state"]

    print("Wczytywanie danych...")
    df = pd.read_csv("data/titanic.csv")

    # Usunięcie kolumn nieprzydatnych do modelowania
    df = df.drop(columns=["name", "ticket", "cabin", "body", "boat", "home.dest"])

    # Uzupełnienie braków
    df["age"] = df["age"].fillna(df["age"].median())
    df["fare"] = df["fare"].fillna(df["fare"].median())
    df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

    # Kodowanie zmiennych kategorycznych
    df["sex"] = df["sex"].map({"male": 0, "female": 1}).astype(int)
    df = pd.get_dummies(df, columns=["embarked"], drop_first=True)
    df["survived"] = df["survived"].astype(int)

    # Podział na zbiory
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )

    train_df.to_csv("data/train.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)
    print(f"Zapisano data/train.csv ({len(train_df)}) i data/test.csv ({len(test_df)})")


if __name__ == "__main__":
    main()
