# Palmer Penguins MLOps Pipeline

Kompletny system MLOps do klasyfikacji gatunków pingwinów. Projekt integruje wersjonowanie danych (DVC), śledzenie eksperymentów (MLflow), optymalizację hiperparametrów (Optuna) oraz serwowanie modelu (BentoML).

## Struktura projektu
```text
├── config/           # Konfiguracja projektu (config.yaml)
├── data/             # Dane (raw, processed)
├── metrics/          # Wyniki ewaluacji modelu (metrics.json)
├── models/           # Wygenerowane artefakty (model.pkl, encoder.pkl)
├── src/              # Kod źródłowy (pipeline, training, preprocessor, service)
├── main.py           # Punkt wejścia do pipeline (opcjonalny)
├── dvc.yaml          # Definicja potoku DVC
└── bentofile.yaml    # Konfiguracja serwowania BentoML
```

## Jak uruchomić projekt

### 1. Przygotowanie środowiska
Zalecane jest użycie środowiska wirtualnego:
```bash
pip install -r requirements.txt
dvc init --subdir
```

### 2. Uruchomienie pipeline'u (DVC)
Cały proces (pobranie danych, przygotowanie, trening, ewaluacja i rejestracja) jest zautomatyzowany przez DVC:
```bash
dvc repro
```
*DVC automatycznie wykryje, które etapy wymagają powtórzenia w oparciu o zmiany w danych lub konfiguracji.*

### 3. Monitoring eksperymentów
Aby podejrzeć przebieg eksperymentów i wyniki w MLflow:
```bash
mlflow ui
```

### 4. Serwowanie modelu (BentoML)
Aby uruchomić serwis predykcyjny lokalnie:
```bash
bentoml serve src.service:PenguinService --reload
```
Następnie otwórz `http://localhost:3000` w przeglądarce, aby przetestować model przez interfejs Swagger UI.

### 5. Budowanie paczki produkcyjnej
Aby zbudować kontener/paczkę do wdrożenia:
```bash
bentoml build
```

## Kluczowe technologie
* **DVC**: Wersjonowanie danych i orkiestracja potoku.
* **MLflow**: Śledzenie eksperymentów i model registry.
* **Optuna**: Automatyczna optymalizacja hiperparametrów Random Forest.
* **BentoML**: Serwowanie modelu przez REST API z walidacją Pydantic.
* **Scikit-Learn**: Przetwarzanie danych i klasyfikacja.

---

*Projekt stworzony w podejściu produkcyjnym (modułowy kod, ścisłe typowanie, separacja konfiguracji).*
