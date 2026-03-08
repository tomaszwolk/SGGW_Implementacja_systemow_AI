"""Moduł treningu i ewaluacji modeli ML."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class ModelTrainer:
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
        model_config = config["model"]
        model_type = model_config["type"]
        model_params = model_config.get("params", {})

        # Tworzymy model na podstawie typu z konfiguracji
        if model_type == "RandomForestClassifier":
            self.model = RandomForestClassifier(**model_params)
        else:
            raise ValueError(f"Nieobsługiwany typ modelu: {model_type}")

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Trenuje model na danych treningowych.

        Argumenty:
            X_train: Macierz cech treningowych.
            y_train: Wektor etykiet treningowych.
        """
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Ewaluuje model na danych testowych i zwraca metryki.

        Oblicza dokładność (accuracy), F1-score, precyzję i czułość (recall).

        Argumenty:
            X_test: Macierz cech testowych.
            y_test: Wektor etykiet testowych.

        Zwraca:
            Słownik z metrykami: accuracy, f1, precision, recall.
        """
        y_pred = self.model.predict(X_test)

        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
        }
