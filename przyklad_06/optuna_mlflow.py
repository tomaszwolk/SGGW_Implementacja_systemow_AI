import mlflow
import mlflow.sklearn
import optuna
import pandas as pd
from mlflow.models import infer_signature
from optuna.integration.mlflow import MLflowCallback
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score, train_test_split, KFold

import matplotlib.pyplot as plt
import optuna.visualization as vis
import seaborn as sns
import os



def load_and_preprocess_data():
    """Pobiera Titanic z OpenML i wykonuje preprocessing."""
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
    X, y = load_and_preprocess_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 50, 300, step=50)
        max_depth = trial.suggest_int("max_depth", 3, 15)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5)

        scores = []
        for step, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
            X_train_cv, X_val_cv = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_train_cv, y_val_cv = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42
            )

            model.fit(X_train_cv, y_train_cv)
            y_pred = model.predict(X_val_cv)
            accuracy = accuracy_score(y_val_cv, y_pred)
            scores.append(accuracy)

            trial.report(accuracy, step)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return sum(scores) / len(scores)

    mlflow_callback = MLflowCallback(
        tracking_uri=mlflow.get_tracking_uri(),
        metric_name="cv_accuracy",
    )
    study = optuna.create_study(direction="maximize", study_name="titanic-optuna", pruner=optuna.pruners.MedianPruner(n_warmup_steps=2))
    study.optimize(objective, n_trials=10, callbacks=[mlflow_callback])

    print(f"Best parameters: {study.best_params}")

    best_model = RandomForestClassifier(**study.best_params, random_state=42)

    with mlflow.start_run(run_name="best-model"):
        mlflow.log_params(study.best_params)

        best_model.fit(X_train, y_train)

        y_pred = best_model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred)

        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_f1_score", test_f1)

        signature = infer_signature(X_train, best_model.predict(X_train))
        mlflow.sklearn.log_model(best_model, "model", signature=signature)

        # wizualizacje
        artifact_dir = "optuna_plots"
        os.makedirs(artifact_dir, exist_ok=True)

        fig = vis.plot_optimization_history(study)
        fig.write_image(f"{artifact_dir}/optimization_history.png")

        fig = vis.plot_contour(study)
        fig.write_image(f"{artifact_dir}/contour.png")

        fig = vis.plot_parallel_coordinate(study)
        fig.write_image(f"{artifact_dir}/parallel_coordinate.png")

        fig = vis.plot_slice(study)
        fig.write_image(f"{artifact_dir}/slice.png")

        mlflow.log_artifacts(artifact_dir)

    print(f"Test accuracy: {test_accuracy}")
    print(f"Test F1 score: {test_f1}")

if __name__ == "__main__":
    main()