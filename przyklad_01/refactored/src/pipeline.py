"""Moduł orkiestrujący cały pipeline ML dla danych Titanic."""

from sklearn.model_selection import train_test_split

from .data_loader import TitanicDataLoader
from .preprocessor import TitanicPreprocessor
from .trainer import ModelTrainer


class TitanicPipeline:
    """Pipeline ML łączący wczytywanie, preprocessing, trening i ewaluację.

    Orkiestruje cały przepływ danych: od pobrania surowych danych
    z OpenML, przez preprocessing, podział na zbiory, trening modelu,
    aż po ewaluację i wypisanie metryk.
    """

    def __init__(self, config: dict):
        """Inicjalizuje pipeline z podaną konfiguracją.

        Tworzy instancje loadera, preprocesora i trainera.

        Argumenty:
            config: Pełny słownik konfiguracji (data, preprocessing,
                    model, training).
        """
        self.config = config
        self.loader = TitanicDataLoader(config)
        self.preprocessor = TitanicPreprocessor(config)
        self.trainer = ModelTrainer(config)

    def run(self) -> dict:
        """Uruchamia cały pipeline ML.

        Kolejność operacji:
        1. Wczytanie danych z OpenML
        2. Podział na cechy (X) i etykietę (y)
        3. Podział na zbiór treningowy i testowy
        4. Dopasowanie preprocesora na danych treningowych
        5. Transformacja obu zbiorów
        6. Trening modelu
        7. Ewaluacja i wypisanie wyników

        Zwraca:
            Słownik z metrykami ewaluacji.
        """
        # Krok 1: Wczytanie danych
        print("Wczytywanie danych...")
        df = self.loader.load()

        # Krok 2: Podział na cechy i etykietę
        X = df.drop(columns=["survived"])
        y = df["survived"].astype(int)

        # Krok 3: Podział na zbiór treningowy i testowy
        training_config = self.config["training"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=training_config["test_size"],
            random_state=training_config["random_state"],
        )

        # Krok 4: Dopasowanie preprocesora na danych treningowych
        print("Preprocessing danych...")
        self.preprocessor.fit(X_train)

        # Krok 5: Transformacja obu zbiorów
        X_train = self.preprocessor.transform(X_train)
        X_test = self.preprocessor.transform(X_test)

        # Krok 6: Trening modelu
        print("Trening modelu...")
        self.trainer.train(X_train, y_train)

        # Krok 7: Ewaluacja
        print("Ewaluacja modelu...")
        metrics = self.trainer.evaluate(X_test, y_test)

        # Wypisanie wyników
        print("\n--- Wyniki ewaluacji ---")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")

        return metrics
 