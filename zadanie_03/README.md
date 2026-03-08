# Zadanie 03 — MLflow Projects: pakowanie skryptu treningowego

## Czas

**20 minut**

## Cel

Opakuj skrypt treningowy w format MLflow Project, tak aby można go było uruchomić jedną komendą z dowolnymi parametrami.

## Wymagania

1. Zmodyfikuj dostarczony `train.py` — dodaj parsowanie argumentów z linii poleceń (`argparse`) dla parametrów: `n_estimators`, `max_depth`, `imputation`.
2. Stwórz plik `MLproject` z entry pointem `main` i trzema parametrami.
3. Stwórz plik `python_env.yaml` ze środowiskiem Python 3.12.
4. Stwórz plik `requirements.txt` z zależnościami.
5. Uruchom projekt:
   ```bash
   mlflow run . --env-manager local -P n_estimators=200 -P max_depth=7
   ```
6. Zweryfikuj wyniki w MLflow UI.

## Plik startowy

Plik `train.py` zawiera kompletny kod treningu i logowania do MLflow, ale z **zahardkodowanymi parametrami**. Twoje zadanie: dodaj `argparse` i użyj wartości z linii poleceń.

## Podpowiedzi

- Wzoruj się na przykładzie `examples/03_mlflow_projects/`.
- Parametry w `MLproject` mają format: `nazwa: {type: typ, default: wartość}`.
- `--env-manager local` pozwala użyć aktywnego środowiska zamiast tworzyć nowe.

## Rozwiązanie

Katalog `rozwiazanie/` zawiera kompletny MLflow Project gotowy do uruchomienia.
