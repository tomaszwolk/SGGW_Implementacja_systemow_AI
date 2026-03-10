import yaml
import argparse
from pathlib import Path
from src.pipeline import PalmerPenguinsPipeline

def main(stage):
    # Ścieżka do pliku konfiguracyjnego
    config_path = Path(__file__).parent / "config" / "config.yaml"

    # Wczytywanie konfiguracji
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    pipeline = PalmerPenguinsPipeline(config)
     # Wykonanie konkretnego etapu
    if stage == "download_data": pipeline.run_download_data()
    elif stage == "prepare_data": pipeline.run_prepare_data()
    elif stage == "train_model": pipeline.run_train_model()
    elif stage == "evaluate_model": pipeline.run_evaluate_model()
    elif stage == "register_bentoml": pipeline.run_register_bentoml()
    elif stage == "all":
        pipeline.run_download_data()
        pipeline.run_prepare_data()
        pipeline.run_train_model()
        pipeline.run_evaluate_model()
        pipeline.run_register_bentoml()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["download_data", "prepare_data", "train_model", "evaluate_model", "register_bentoml", "all"])
    args = parser.parse_args()
    main(args.stage)
    