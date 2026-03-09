import numpy as np
import mlflow
import joblib
import optuna
from mlflow.models import infer_signature
from optuna.integration.mlflow import MLflowCallback
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score

class PenguinTrainer:
    """Klasa odpowiedzialna za trening i ewaluację modelu klasyfikacji.

    Tworzy model na podstawie konfiguracji, trenuje go
    i oblicza metryki jakości.
    """

    def __init__(self, config: dict):
        """Inicjalizuje trainer na podstawie konfiguracji modelu.

        Argumenty:
            config: Słownik z kluczem 'model' zawierającym
                    'type' (typ modelu) oraz 'params' (hiperparametry).
        """
        self.model_config = config["model"]
        self.model_type = self.model_config["type"]
        self.train_params = self.model_config["train_params"]
        self.best_model = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Trenuje model na danych treningowych.

        Argumenty:
            X_train: Macierz cech treningowych.
            y_train: Wektor etykiet treningowych.
        """
        def objective(trial):
            n_estimators = trial.suggest_int(*self.train_params["n_estimators"])
            max_depth = trial.suggest_int(*self.train_params["max_depth"])
            min_samples_split = trial.suggest_int(*self.train_params["min_samples_split"])
            min_samples_leaf = trial.suggest_int(*self.train_params["min_samples_leaf"])
            random_state = self.train_params["random_state"]

            # Tworzymy model na podstawie typu konfiguracji
            if self.model_type == "RandomForestClassifier":
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    random_state=random_state
                )
            else:
                raise ValueError(f"Nieobsługiwany typ modelu: {self.model_type}")
            
            scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
            return scores.mean()
        
        mlflow.set_experiment(self.model_config["experiment_name"])

        mlflow_callback = MLflowCallback(
            tracking_uri=mlflow.get_tracking_uri(),
            metric_name="cv_accuracy"
        )

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.train_params["n_trials"], callbacks=[mlflow_callback])

        print(f"\nNajlepsze parametry: {study.best_params}")
        print(f"Najlepsza CV accuracy: {study.best_value:.4f}")

        self.best_model = RandomForestClassifier(**study.best_params, random_state=self.train_params["random_state"])
        self.best_model.fit(X_train, y_train)

        return self

    def evaluate_and_log(self, model, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
        y_pred = model.predict(X_test)
        metrics = {
            "test_accuracy": round(accuracy_score(y_test, y_pred), 4),
            "test_f1_score": round(f1_score(y_test, y_pred, average="weighted"), 4)

        }

        print(f"Metrics: {metrics}")

        signature = infer_signature(X_test, y_pred) # Tworzenie sygnatury

        mlflow.set_experiment(self.model_config["experiment_name"])
        with mlflow.start_run(run_name="best-model"):
            mlflow.log_metrics(metrics)
            mlflow.log_params(model.get_params()) 

            # Logujemy model z sygnaturą
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                signature=signature
            )

        return metrics

    def save_model(self, filepath: str): # DELETE?
        """Zapisuje wyuczony model do pliku .pkl"""
        joblib.dump(self.best_model, filepath)

    def log_best_model(self, model, X_test: np.ndarray, y_test: np.ndarray):
        y_pred = model.predict(X_test)
        signature = infer_signature(X_test, y_pred) # Tworzenie sygnatury

        # Logowanie do tego samego eksperymentu
        mlflow.set_experiment(self.model_config["experiment_name"])
        
        with mlflow.start_run(run_name="best-model-final"):
            mlflow.log_params(model.get_params()) 
            
            # Logujemy model z sygnaturą
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                signature=signature
            )