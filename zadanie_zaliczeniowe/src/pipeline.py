import json
import joblib
import pandas as pd
import bentoml
import mlflow

from pathlib import Path
from sklearn.model_selection import train_test_split

from .data_loader import PalmerPenguinsDataLoader
from .preprocessor import PenguinPreprocessor
from .trainer import PenguinTrainer


class PalmerPenguinsPipeline:

    def __init__(self, config: dict):
        self.config = config
        self.loader = PalmerPenguinsDataLoader(config)
        self.project_root = Path(__file__).parent.parent
        self.target = self.config["preprocessing"]["target_col"]

    def _get_full_path(self, rel_path: str) -> Path:
        full_path = self.project_root / rel_path
        # Tworzymy strukturę katalogów jeśli nie istnieje
        full_path.parent.mkdir(parents=True, exist_ok=True)

        return full_path

    def _save_data(self, df: pd.DataFrame, config_path_key: str) -> Path:
        """Prywatna metoda pomocnicza do zapisu DataFrame."""
        # Pobieramy ścieżkę
        rel_path = self.config["data"][config_path_key]
        full_path = self._get_full_path(rel_path)

        # Zapis pliku
        df.to_csv(full_path, index=False)
        return full_path

    def _to_json(self, data, config_path_key: str) -> Path:
        """Prywatna metoda pomocnicza do zapisu do json."""
        # Pobieramy ścieżkę
        rel_path = self.config["tests"][config_path_key]
        full_path = self._get_full_path(rel_path)

        # Zapis pliku
        with open(full_path, "w") as file:
            json.dump(data, file, indent=2)

        return full_path

    def _save_model(self, model: PenguinPreprocessor | PenguinTrainer, config_path_key: str) -> Path:
        """Prywatna metoda pomocnicza do zapisu encodera."""
        # Pobieramy ścieżkę
        if config_path_key in self.config["preprocessing"]:
            rel_path = self.config["preprocessing"]["encoder_path"]
        else:
            rel_path = self.config["model"]["rf_model_path"]
        full_path = self._get_full_path(rel_path)

        joblib.dump(model, full_path)

        return full_path

    def _split_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame]:
        target_col = self.config["preprocessing"]["target_col"]

        # Mapowanie targetu
        mapping = self.config["data"]["target_mapping"]
        df[target_col] = df[target_col].str.strip().map(mapping).astype(int)

        # Podział
        X = df.drop(columns=[target_col])
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
            )

        return X_train, X_test, y_train, y_test

    def run_download_data(self):
        print("Wczytywanie danych")
        df_raw = self.loader.load()
        print("Poprawnie wczytano dane")

        print("Zapis danych na dysku")
        saved_path = self._save_data(df_raw, "raw_data_path")
        print(f"Dane zapisane do: {saved_path}")
        

    def run_prepare_data(self):
        print("Preprocessing i zapis encodera.")
        full_raw_data_path = self._get_full_path(self.config["data"]["raw_data_path"])
        df_raw = pd.read_csv(full_raw_data_path)
        df = df_raw.dropna().copy()

        print("Split danych na train / test.")
        X_train, X_test, y_train, y_test = self._split_data(df)

        preprocessor = PenguinPreprocessor(
            categorical_cols=self.config["preprocessing"]["categorical_cols"]
            )
        preprocessor.fit(X_train)
        encoder_path = self._save_model(preprocessor, "encoder_path")
        print(f"Encoder zapisany w: {encoder_path}")

        X_train_trans = preprocessor.transform(X_train)
        X_test_trans = preprocessor.transform(X_test)

        # Zapis
        X_train_trans_saved_path = self._save_data(pd.concat([X_train_trans, y_train], axis=1), "train_data_path")
        X_test_trans_saved_path = self._save_data(pd.concat([X_test_trans, y_test], axis=1), "test_data_path")
        print(f"Dane przeprocesowane zapisane do: \n\t\t{X_train_trans_saved_path}\n\t\t{X_test_trans_saved_path}")

    def run_train_model(self):
        # Wczytanie danych treningowych
        full_train_data_path = self._get_full_path(self.config["data"]["train_data_path"])
        df = pd.read_csv(full_train_data_path)
        X_train = df.drop(columns=self.target)
        y_train = df[self.target]

        # Inicjalizacja modelu
        trainer = PenguinTrainer(self.config)
        trainer.train(X_train, y_train)
        trainer_path = self._save_model(trainer.best_model, "rf_model_path")
        print(f"Model zapisany w: {trainer_path}")

    def run_evaluate_model(self):
        # Wczytanie danych
        full_test_path = self._get_full_path(self.config["data"]["test_data_path"])
        df_test = pd.read_csv(full_test_path)
        X_test = df_test.drop(columns=self.target)
        y_test = df_test[self.target]

        # Wczytanie modelu z pliku
        model_path = self._get_full_path(self.config["model"]["rf_model_path"])
        model = joblib.load(model_path)
        print(f"Wczytano model do ewaluacji z: {model_path}")

        # Ewaluacja modelu
        trainer = PenguinTrainer(self.config)
        metrics = trainer.evaluate_and_log(model, X_test, y_test)

        # Zapis metryk do .json
        full_metrics_path = self._to_json(metrics, "metrics_path")
        print(f"Metryki zapisane w: {full_metrics_path}")

    def run_register_bentoml(self):
        # Wczytywanie modelu i enkoder z plików .pkl
        model_path = self._get_full_path(self.config["model"]["rf_model_path"])
        encoder_path = self._get_full_path(self.config["preprocessing"]["encoder_path"])

        model = joblib.load(model_path)
        encoder = joblib.load(encoder_path)

        # Zapis modeli do Model Store BentoML
        bento_model = bentoml.sklearn.save_model("penguins_classifier", model)
        bento_encoder = bentoml.sklearn.save_model("penguins_encoder", encoder.encoder)

        print(f"Zarejestrowano model: {bento_model}")
        print(f"Zarejestrowano encoder: {bento_encoder}")

        # Dodanie logów do MLflow o rejestracji
        mlflow.set_experiment(self.config["model"]["experiment_name"])
        with mlflow.start_run(run_name="bentoml-registration"):
            mlflow.log_param("bento_model_tag", str(bento_model.tag))
            mlflow.log_param("bento_encoder_tag", str(bento_encoder.tag))
