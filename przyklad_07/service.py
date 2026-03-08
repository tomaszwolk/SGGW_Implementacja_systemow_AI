import bentoml
from bentoml.models import BentoModel
from pydantic import BaseModel
import numpy as np
import pandas as pd


class TitanicFeatures(BaseModel):
    age: float
    fare: float
    sex: int
    pclass: float
    sibsp: float
    parch: float
    embarked_Q: int = 0
    embarked_S: int = 0

@bentoml.service(name="titanic_classifier_service")
class TitanicClassifierService:
    titanic_model = BentoModel("titanic_classifier:latest")

    def __init__(self):
        self.model = bentoml.sklearn.load_model(self.titanic_model)

    @bentoml.api()
    def predict(self, features: TitanicFeatures):
        params = np.array([[features.age, features.fare, features.sex, features.pclass, features.sibsp, features.parch, features.embarked_Q, features.embarked_S]])

        pred = self.model.predict(params)
        return {"prediction": int(pred[0])}

    @bentoml.api()
    def predict_batch(self, features_batch: pd.DataFrame):
        predictions = self.model.predict(features_batch)

        results = []
        for pred in predictions:
            results.append({"prediction": int(pred[0])})
        return results