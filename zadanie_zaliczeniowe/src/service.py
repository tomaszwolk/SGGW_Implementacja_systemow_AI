import bentoml
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from src.preprocessor import PenguinPreprocessor

class PenguinFeatures(BaseModel):
    island: str = Field(..., description="Wyspa: Biscoe, Dream, Torgersen")
    culmen_length_mm: float = Field(..., description="Długość dzioba w mm")
    culmen_depth_mm: float = Field(..., description="Głębokość dzioba w mm")
    flipper_length_mm: float = Field(..., description="Długość płetwy w mm")
    body_mass_g: float = Field(..., description="Masa ciała w gramach")
    sex: str = Field(..., description="Płeć: MALE, FEMALE")


@bentoml.service(name="penguin_service")
class PenguinService:
    rf_model = bentoml.models.BentoModel("penguins_classifier:latest")
    encoder_model = bentoml.models.BentoModel("penguins_encoder:latest")

    def __init__(self):
        self.model = bentoml.sklearn.load_model(self.rf_model)
        self.encoder = bentoml.sklearn.load_model(self.encoder_model)
        # Mapowanie odwrotne by API zwracało nawzy zamiast int
        self.target_mapping_inv: dict = {0: "Adelie", 1: "Chinstrap", 2: "Gentoo"}

    @bentoml.api()
    def predict(self, features: PenguinFeatures):
        # Konwersja danych wejściowych. 
        # .model_dump() zamienia obiekt Pydantic na słwonik Pythona, a pandas na DataFrame
        df_input = pd.DataFrame([features.model_dump()])

        # Kolejność i wybór kolumn zgodny z trenowaniem preprocesora
        categorical_cols = ["island", "sex"]
        numeric_cols = ["culmen_length_mm", "culmen_depth_mm", "flipper_length_mm", "body_mass_g"]

        # Uporządkuj wejście
        df_input = df_input[categorical_cols + numeric_cols]

        # Preprocessing z użyciem encodera – tak jak w PenguinPreprocessor.transform
        encoded_array = self.encoder.transform(df_input[categorical_cols])
        df_encoded = pd.DataFrame(
            encoded_array,
            columns=self.encoder.get_feature_names_out(),
            index=df_input.index,
        )
        df_num = df_input[numeric_cols]
        df_transformed = pd.concat([df_num, df_encoded], axis=1)
            
        # Predykcja
        prediction = int(self.model.predict(df_transformed)[0])
        probas = self.model.predict_proba(df_transformed)[0]
        confidence = float(probas[prediction])

        # Wynik
        species_name = self.target_mapping_inv.get(prediction, "Unknown")
        return {
            "prediction": prediction,
            "species": species_name,
            "confidence": round(confidence, 4),
        }
    