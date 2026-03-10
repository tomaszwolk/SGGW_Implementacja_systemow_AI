import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder


class PenguinPreprocessor:
    """Klasa do wstępnego przetwarzania danych Palmer Penguins."""

    def __init__(self, categorical_cols: list = None):
        self.categorical_cols = categorical_cols or ["sex", "island"]
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self._is_fitted = False

    def fit(self, X: pd.DataFrame, y=None):
        """Uczy się parametrów transformacji na podstawie danych treningowych.

        Argumenty:
            df: DataFrame z danymi treningowymi.

        Zwraca:
            Referencja do self.
        """
        # Fit encoder na danych kategorycznych
        self.encoder.fit(X[self.categorical_cols])

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError(
                "Preprocesor nie został dopasowany. Wywołaj fit() przed transform()."
            )
        
        # Tworzymy kopię by nie nadpisywać oryginalnego DataFrame
        X_clean = X.copy()

        # Transformowanie kolumn kategorycznych
        encoded_array = self.encoder.transform(X_clean[self.categorical_cols])

        # Tworzenie końcowego DataFrame
        df_encoded = pd.DataFrame(
            encoded_array,
            columns=self.encoder.get_feature_names_out(),
            index=X_clean.index
            )
        df_num = X_clean.drop(columns=self.categorical_cols)
        df_clean_encoded = pd.concat([df_num, df_encoded], axis=1)

        return df_clean_encoded

    def load_encoder(self, filepath: str):
        """Wczytuje encoder z pliku .pkl."""
        self.encoder = joblib.load(filepath)
        self._is_fitted = True
