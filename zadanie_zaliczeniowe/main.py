import yaml
from pathlib import Path
from src.pipeline import PalmerPenguinsPipeline

def main():
    # Ścieżka do pliku konfiguracyjnego
    config_path = Path(__file__).parent / "config" / "config.yaml"

    # Wczytywanie konfiguracji
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    pipeline = PalmerPenguinsPipeline(config)
    pipeline.run_download_data()
    pipeline.run_prepare_data()
    pipeline.run_train_model()
    pipeline.run_evaluate_model()
    pipeline.run_register_bentoml()

if __name__ == "__main__":
    main()
