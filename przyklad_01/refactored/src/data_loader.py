"""Moduł odpowiedzialny za wczytywanie danych Titanic z OpenML."""

import pandas as pd
from sklearn.datasets import fetch_openml


class TitanicDataLoader:
    """Klasa do wczytywania i wstępnego czyszczenia danych Titanic.

    Pobiera dane z repozytorium OpenML i usuwa kolumny
    wskazane w konfiguracji.
    """

    def __init__(self, config: dict):
        """Inicjalizuje loader z podaną konfiguracją.

        Argumenty:
            config: Słownik z kluczem 'data' zawierającym
                    'dataset_id' oraz 'drop_columns'.
        """
        self.config = config

    def load(self) -> pd.DataFrame:
        """Wczytuje dane z OpenML i usuwa zbędne kolumny.

        Zwraca:
            DataFrame z danymi Titanic po usunięciu
            kolumn wskazanych w konfiguracji.
        """
        data_config = self.config["data"]
        dataset_id = data_config["dataset_id"]

        # Pobieramy zbiór danych z OpenML
        data = fetch_openml(data_id=dataset_id, as_frame=True)
        df = data.frame

        # Usuwamy kolumny wskazane w konfiguracji
        drop_columns = data_config.get("drop_columns", [])
        df = df.drop(columns=drop_columns)

        return df
