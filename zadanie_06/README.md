# Zadanie 06 — BentoML: serwis z dwoma modelami

## Czas

**30 minut**

## Cel

Zbuduj serwis BentoML obsługujący dwa modele (RandomForest i LogisticRegression) z osobnymi endpointami. Użytkownik API może wybrać, którego modelu chce użyć.

## Wymagania

1. Wytrenuj **2 modele** na zbiorze Titanic (OpenML, id: 40945):
   - `RandomForestClassifier`
   - `LogisticRegression`
2. Zapisz oba modele przez `bentoml.sklearn.save_model` pod różnymi nazwami:
   - `titanic_rf`
   - `titanic_logreg`
3. Zdefiniuj serwis BentoML z:
   - `TitanicFeatures` — model Pydantic (ten sam co w przykładzie)
   - **`predict_rf`** — endpoint używający RandomForest
   - **`predict_logreg`** — endpoint używający LogisticRegression
   - Oba zwracają: `{"model": "...", "prediction": 0/1, "confidence": float}`
   (confidence = self.model.predict_proba(params)[0])
4. Uruchom serwis i przetestuj oba endpointy w Swagger UI.

## Uruchomienie

```bash
cd SGGW/assignments/06_bentoml/rozwiazanie
source ../../.venv/bin/activate

# 1. Wytrenuj i zapisz modele
python train_models.py

# 2. Uruchom serwis
bentoml serve service:TitanicTwoModelsService --reload

# 3. Otwórz Swagger UI
# → http://localhost:3000/docs
```

## Podpowiedzi

- Wzoruj się na przykładzie `examples/06_bentoml/`.
- Każdy model wymaga osobnego `bentoml.models.BentoModel(...)` w klasie serwisu.
- W `__init__` załaduj oba modele przez `bentoml.sklearn.load_model(...)`.
- Porównaj odpowiedzi obu modeli na tych samych danych wejściowych.

## Rozwiązanie

`rozwiazanie/`
