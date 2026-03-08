import bentoml
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def load_and_preprocess_titanic():
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
    X, y = load_and_preprocess_titanic()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)

    model_tag = bentoml.sklearn.save_model(
        "titanic_classifier",
        model,
        signatures={
            "predict": {"batchable": True, "batch_dim": 0},
            "predict_proba": {"batchable": True, "batch_dim": 0},
        },
        metadata={"accuracy": accuracy},
    )
    print(f"Model zapisany: {model_tag}")

if __name__ == "__main__":
    main()