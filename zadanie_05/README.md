# Zadanie 05 — Optuna + MLflow: optymalizacja hiperparametrów

## Czas

**30 minut**

## Cel

Użyj Optuna do znalezienia optymalnych hiperparametrów RandomForest na zbiorze Titanic. Śledź wszystkie próby w MLflow za pomocą MLflowCallback.

## Wymagania

1. Zdefiniuj **funkcję celu** (`objective`) z co najmniej 3 hiperparametrami:
   - `n_estimators` (trial.suggest_int)
   - `max_depth` (trial.suggest_int)
   - `min_samples_split` (trial.suggest_int)
2. Ewaluuj model **walidacją krzyżową** (5-fold `cross_val_score`).
3. Użyj **MLflowCallback** do automatycznego śledzenia prób.
4. Uruchom co najmniej **15 prób**.
5. Po optymalizacji: wytrenuj model z najlepszymi parametrami na pełnym zbiorze treningowym.
6. Zaloguj najlepszy model do MLflow z `log_params`, `log_metrics`, `log_model`.

## Plik startowy

Plik `starter.py` zawiera gotowy preprocessing i szkielet z TODO.

## Uruchomienie

```bash
cd SGGW
source .venv/bin/activate
python assignments/05_optuna_mlflow/starter.py
mlflow ui
# → http://localhost:5000
```

## Podpowiedzi

- Wzoruj się na przykładzie `examples/07_optuna/`.
- `MLflowCallback` automatycznie loguje parametry i metrykę każdej próby.
- `study.best_params` zwraca słownik najlepszych parametrów.
- `study.best_value` zwraca najlepszą wartość metryki.

## Rozwiązanie

`rozwiazanie/solution.py`
