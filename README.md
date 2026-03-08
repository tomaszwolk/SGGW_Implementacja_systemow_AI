# SGGW Implementacja Systemów AI

Repozytorium zawiera przykłady oraz zadania z przedmiotu Implementacja Systemów AI z kierunku Inżynieria AI na SGGW.

Zakres przedmiotu:
- MLFlow
- DVC
- Optuna
- BentoML

## UV + Linux/MacOs

uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

## Venv + Windows

python.exe -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt

python.exe -m pip install -r requirements.txt

## Lista komend

### MLFLOw

| Komenda | Opis |
|---------|------|
| `mlflow ui` | Uruchamia interfejs webowy MLflow |
| `mlflow run .` | Uruchamia projekt MLflow z bieżącego katalogu |
| `mlflow run . -P alpha=0.5` | Uruchamia projekt z parametrem |


### DVC

| Komenda | Opis |
|---------|------|
| `dvc init` | Inicjalizuje DVC w projekcie |
| `dvc add data/dataset.csv` | Dodaje plik do śledzenia przez DVC |
| `dvc remote add -d myremote s3://bucket/path` | Dodaje zdalne repozytorium |
| `dvc remote add -d myremote /Users/...` | Dodaje zdalne repozytorium (lokalny katalog) |
| `dvc remote add -d myremote gdrive://folder_id` | Dodaje Google Drive jako zdalne repozytorium |
| `dvc push` | Wysyła dane do zdalnego repozytorium |
| `dvc pull` | Pobiera dane ze zdalnego repozytorium |
| `dvc checkout` | Przywraca dane z zdalnego repozytorium |
| `dvc du . ` | Pokazuje rozmiar danych w projekcie |
| `dvc gc --workspace` | Usuwa niepotrzebne dane z cache |
| `dvc repro` | Uruchamia ponownie pipeline |
| `dvc dag` | Pokazuje graf zależności |
| `dvc metrics show` | Pokazuje metryki |
| `dvc metrics diff` | Porównuje metryki |
| `dvc exp run --set-param n_estimators=200` | Uruchamia eksperyment z modyfikacją parametrów |
| `dvc exp show` | Pokazuje eksperymenty |

## BentoML

| Komenda | Opis |
|---------|------|
| `bentoml models list` | Pokazuje modele |
| `bentoml serve service:NazwaSerwisu` | Uruchamia serwis |
| `bentoml build` | Buduje serwis |
| `bentoml containerize` | Buduje kontener |

## DVC Workflow

1. Tworzymy plik w katalogu data (nowy!)
2. Dodajemy do dvc: `dvc add nazwa_pliku`
3. dvc push, git add, git commit
4. Zmiana zawartości pliku (tego nowego)
5. dvc add, dvc push, git add, git commit
6. git checkout [nazwa brancha lub id commitu] (na poprzedni stan pliku)
7. dvc checkout
8. git checkout [nazwa brancha]
9. Usuwamy pliki śledzone przez dvc z katalogu data oraz .dvc/cache
10. dvc pull

https://mikulskibartosz.github.io/sggw_implementacja_ai/#/3/10