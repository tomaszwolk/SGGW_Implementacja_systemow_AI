"""Moduł preprocessingu danych Titanic w stylu sklearn (fit/transform)."""

import pandas as pd


class TitanicPreprocessor:
    """Preprocesor danych Titanic zgodny z interfejsem sklearn.

    Uczy się parametrów transformacji (mediany, mody) na zbiorze
    treningowym, a następnie stosuje je na dowolnym zbiorze danych.
    Dzięki temu unikamy wycieku danych (data leakage).
    """

    def __init__(self, config: dict):
        """Inicjalizuje preprocesor z podaną konfiguracją.

        Argumenty:
            config: Słownik z kluczem 'preprocessing' zawierającym
                    strategie uzupełniania braków i kodowania zmiennych.
        """
        self.config = config
        self._medians = {}
        self._modes = {}
        self._is_fitted = False

    def fit(self, df: pd.DataFrame) -> "TitanicPreprocessor":
        """Uczy się parametrów transformacji na podstawie danych treningowych.

        Wyznacza mediany i mody dla kolumn wskazanych w konfiguracji.

        Argumenty:
            df: DataFrame z danymi treningowymi.

        Zwraca:
            Referencja do self (umożliwia łańcuchowe wywołania).
        """
        fill_strategy = self.config["preprocessing"]["fill_strategy"]

        for column, strategy in fill_strategy.items():
            if strategy == "median":
                self._medians[column] = df[column].median()
            elif strategy == "mode":
                self._modes[column] = df[column].mode()[0]

        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Stosuje wyuczone transformacje na podanym zbiorze danych.

        Uzupełnia braki, koduje zmienne kategoryczne (sex -> 0/1,
        embarked -> one-hot encoding).

        Argumenty:
            df: DataFrame do przekształcenia.

        Zwraca:
            Przekształcony DataFrame.

        Wyjątki:
            RuntimeError: Jeśli transform() wywołano przed fit().
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Preprocesor nie został dopasowany. Wywołaj fit() przed transform()."
            )

        df = df.copy()

        # Uzupełniamy braki wyuczonymi wartościami
        for column, median_val in self._medians.items():
            df[column] = df[column].fillna(median_val)

        for column, mode_val in self._modes.items():
            df[column] = df[column].fillna(mode_val)

        # Kodujemy zmienną sex na podstawie mapowania z konfiguracji
        encode_config = self.config["preprocessing"]["encode"]
        if "sex" in encode_config:
            df["sex"] = df["sex"].map(encode_config["sex"]).astype(int)

        # One-hot encoding dla embarked
        if encode_config.get("embarked") == "onehot":
            df = pd.get_dummies(df, columns=["embarked"], drop_first=True)

        return df
