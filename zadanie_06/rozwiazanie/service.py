"""Serwis BentoML z dwoma modelami: RandomForest i LogisticRegression."""

import bentoml
import numpy as np
from pydantic import BaseModel


class TitanicFeatures(BaseModel):
    age: float
    fare: float
    sex: int
    pclass: float
    sibsp: float
    parch: float
    embarked_Q: int = 0
    embarked_S: int = 0


@bentoml.service(name="titanic_two_models_service")
class TitanicTwoModelsService:
    rf_model = bentoml.models.BentoModel("titanic_rf:latest")
    logreg_model = bentoml.models.BentoModel("titanic_logreg:latest")

    def __init__(self):
        self.rf = bentoml.sklearn.load_model(self.rf_model)
        self.logreg = bentoml.sklearn.load_model(self.logreg_model)

    def _predict(self, model, model_name, features):
        arr = np.array([[
            features.age, features.fare, features.sex, features.pclass,
            features.sibsp, features.parch, features.embarked_Q, features.embarked_S,
        ]])
        prediction = int(model.predict(arr)[0])
        probas = model.predict_proba(arr)[0]
        confidence = float(probas[prediction])
        return {
            "model": model_name,
            "prediction": prediction,
            "confidence": round(confidence, 4),
        }

    @bentoml.api()
    def predict_rf(self, features: TitanicFeatures) -> dict:
        return self._predict(self.rf, "RandomForest", features)

    @bentoml.api()
    def predict_logreg(self, features: TitanicFeatures) -> dict:
        return self._predict(self.logreg, "LogisticRegression", features)
