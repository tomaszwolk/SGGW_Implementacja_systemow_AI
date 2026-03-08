# Zadanie 04 — Pipeline DVC z 4 etapami

## Czas

**45 minut**

## Cel

Zbuduj reprodukowalny, 4-etapowy pipeline DVC: download → prepare → train → **evaluate**. Trzy pierwsze skrypty są gotowe — Twoim zadaniem jest napisać `dvc.yaml` i skrypt ewaluacji.

## Wymagania

1. **Stwórz plik `dvc.yaml`** z 4 etapami:
   - `download_data` — pobiera dane (deps: skrypt, params: data.dataset_id, outs: data/titanic.csv)
   - `prepare_data` — preprocessing (deps: skrypt + dane, params: prepare.*, outs: data/train.csv + data/test.csv)
   - `train_model` — trening (deps: skrypt + dane, params: model.*, outs: models/model.pkl)
   - `evaluate` — ewaluacja (deps: skrypt + model + dane testowe, metrics: metrics.json z cache: false)

2. **Napisz skrypt `src/evaluate.py`** — wczytuje model i dane testowe, oblicza accuracy i F1, zapisuje do `metrics.json`.

3. **Uruchom pipeline**: `dvc repro`

4. **Zmień hiperparametr** w `params.yaml` (np. `n_estimators: 200`) i uruchom ponownie — sprawdź, że `download_data` i `prepare_data` się nie przebudowują.

5. **Porównaj metryki**: `dvc metrics diff`

## Dostarczone pliki

- `params.yaml` — konfiguracja (gotowy)
- `src/download_data.py` — pobieranie danych (gotowy)
- `src/prepare_data.py` — preprocessing (gotowy)
- `src/train_model.py` — trening modelu (gotowy, **bez metryk** — metryki liczy evaluate)
- `src/evaluate.py` — **szkielet z TODO** (do uzupełnienia)

## Uruchomienie

```bash
cd SGGW/assignments/04_dvc_pipeline
dvc init
dvc repro
dvc dag
dvc metrics show

# Zmień parametr i porównaj
# (edytuj params.yaml)
dvc repro
dvc metrics diff
```

## Podpowiedzi

- Wzoruj się na przykładzie `examples/04_dvc_pipeline/`.
- Użyj `dvc dag` aby zobaczyć graf zależności.
- Etap `evaluate` zależy od modelu (`models/model.pkl`) i danych testowych (`data/test.csv`).

## Rozwiązanie

`rozwiazanie/dvc.yaml` i `rozwiazanie/src/evaluate.py`
