import pandas as pd
from sklearn.datasets import fetch_openml

class PalmerPenguinsDataLoader:
    """Klasa do wczytywania danych z repozytorium OpenML."""

    def __init__(self, config: dict):
        """Inicjalizacja loadera."""
        self.config = config

    def load(self) -> pd.DataFrame:
        data_config = self.config["data"]
        dataset_id = data_config["dataset_id"]

        data_raw = fetch_openml(data_id=dataset_id, as_frame=True)
        df = data_raw.frame

        return df