"""Punkt wejściowy pipeline'u ML dla danych Titanic.

Wczytuje konfigurację z pliku YAML, tworzy pipeline i uruchamia go.
"""

from pathlib import Path

import yaml

from src.pipeline import TitanicPipeline


def main():
    """Główna funkcja uruchamiająca pipeline klasyfikacji Titanic."""
    # Ścieżka do pliku konfiguracyjnego
    config_path = Path(__file__).parent / "config" / "config.yaml"

    # Wczytanie konfiguracji z pliku YAML
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Utworzenie i uruchomienie pipeline'u
    pipeline = TitanicPipeline(config)
    metrics = pipeline.run()


if __name__ == "__main__":
    main()
 