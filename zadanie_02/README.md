# Zadanie 02 — MLflow Tracking: porównanie preprocessingu

## Czas

**30 minut**

## Cel

Porównaj wpływ dwóch strategii imputacji (`median` vs `mean` dla kolumny `age`) na wyniki klasyfikacji Titanic. Każdy wariant zaloguj do MLflow z pełnym zestawem metryk, artefaktów i tagów.

## Wymagania

1. Użyj zbioru Titanic z OpenML (id: 40945).
2. Wytrenuj **ten sam model** (`RandomForestClassifier`, te same hiperparametry) z dwoma wariantami preprocessingu:
   - Wariant A: `age` uzupełniony **medianą**
   - Wariant B: `age` uzupełniony **średnią**
3. Dla każdego wariantu zaloguj do MLflow:
   - **Parametry** (`log_params`): hiperparametry modelu + strategia imputacji
   - **Metryki** (`log_metrics`): accuracy, F1, precision, recall
   - **Model** z sygnaturą (`log_model` + `infer_signature`)
   - **Artefakty** (`log_artifact`):
     - Macierz pomyłek (PNG)
     - Krzywa ROC (PNG)
   - **Tag** (`set_tag`): `imputation=median` / `imputation=mean`
4. Porównaj oba warianty w MLflow UI.

## Plik startowy

Plik `starter.py` zawiera gotowy kod preprocessingu i treningu. Twoje zadanie to uzupełnić **8 miejsc oznaczonych `TODO`** — wszystkie dotyczą integracji z MLflow.

## Uruchomienie

```bash
cd SGGW
source .venv/bin/activate
python zadanie_02/starter.py
mlflow ui
# → http://localhost:5000
```

## Podpowiedzi

- Wzoruj się na przykładach z `examples/02_mlflow_tracking/`.
- Do krzywej ROC użyj `sklearn.metrics.roc_curve` i `sklearn.metrics.auc`.
- Do macierzy pomyłek użyj `ConfusionMatrixDisplay`.
- Wykresy zapisz do plików tymczasowych, zaloguj przez `mlflow.log_artifact()`, potem usuń pliki.

## Rozwiązanie

`rozwiazanie/solution.py`
