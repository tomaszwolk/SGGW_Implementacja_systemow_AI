# Słownik terminów ML i MLOps

## Podstawy ML

**Model** — wyuczony algorytm, który na podstawie danych wejściowych generuje predykcje lub klasyfikacje.

**Pipeline / Potok ML** — sekwencja kroków przetwarzania danych i modelowania, gdzie wyjście jednego etapu staje się wejściem następnego.

**Preprocessing** — wstępne przetwarzanie danych przed treningiem modelu, np. czyszczenie, normalizacja, imputacja braków.

**Feature Engineering** — proces tworzenia, wybierania i transformacji cech (zmiennych) w celu poprawy jakości modelu.

**Training / Trening** — proces uczenia modelu na danych treningowych poprzez optymalizację parametrów.

**Inference** — faza wykorzystania wytrenowanego modelu do generowania predykcji na nowych danych.

**Evaluation / Ewaluacja** — ocena jakości modelu na zbiorze testowym przy użyciu odpowiednich metryk.

**Predykcja / Prediction** — wynik działania modelu — przewidywana wartość lub klasa dla danych wejściowych.

**Klasyfikacja binarna** — problem uczenia maszynowego z dwoma możliwymi klasami wynikowymi (np. przeżył/nie przeżył).

**Klasyfikacja wieloklasowa** — problem uczenia maszynowego z więcej niż dwoma klasami wynikowymi (np. 3 gatunki pingwinów).

---

## Wzorzec scikit-learn

**fit()** — metoda ucząca model lub transformer na danych treningowych, zapisująca nauczone parametry.

**transform()** — metoda stosująca nauczone przekształcenie do danych (np. imputacja, skalowanie).

**predict()** — metoda generująca predykcje modelu dla nowych danych wejściowych.

---

## Walidacja modelu

**Cross-validation / Walidacja krzyżowa** — technika oceny modelu polegająca na wielokrotnym podziale danych na zbiory treningowe i walidacyjne.

**k-fold** — wariant walidacji krzyżowej, w którym dane dzielone są na k równych części; model trenowany jest k razy, za każdym razem na k-1 częściach.

---

## Optymalizacja hiperparametrów

**Hiperparametry** — parametry modelu ustawiane przed treningiem (np. liczba drzew, głębokość), których model nie uczy się z danych.

**Grid Search** — metoda przeszukiwania wszystkich kombinacji hiperparametrów z zadanej siatki wartości.

**Random Search** — metoda losowego próbkowania hiperparametrów z zadanych rozkładów; często skuteczniejsza niż Grid Search.

**Optymalizacja bayesowska** — metoda inteligentnego doboru hiperparametrów, która uczy się z poprzednich prób i przewiduje najlepsze wartości.

**Pruning** — technika wczesnego kończenia słabo rokujących prób optymalizacji w celu oszczędności czasu.

**Trial** — pojedyncza próba w procesie optymalizacji hiperparametrów z konkretnym zestawem wartości.

**Study** — seria prób (trials) optymalizujących jeden cel (minimalizacja lub maksymalizacja metryki).

**Objective / Funkcja celu** — funkcja zwracająca wartość metryki do optymalizacji; przyjmuje trial i zwraca wynik.

**Sampler (TPE)** - strategia wybierania kolejnych wartości hiperparametrów

---

## Narzędzia MLOps

### MLflow

**MLflow Tracking** — komponent MLflow do śledzenia parametrów, metryk i artefaktów eksperymentów.

**MLflow Projects** — format pakowania kodu ML z definicją środowiska i parametryzowanymi punktami wejścia.

**MLflow Models** — standardowy format pakowania modeli ML umożliwiający deployment w różnych środowiskach.

**MLflow Model Registry** — rejestr modeli do zarządzania wersjami, etapami (staging/production) i metadanymi.

**Experiment** — logiczna grupa runów w MLflow, np. eksperymenty z różnymi algorytmami na tym samym zbiorze danych.

**Run** — pojedyncze wykonanie eksperymentu z konkretnymi parametrami, metrykami i artefaktami.

**log_param** — funkcja MLflow zapisująca wartość parametru (hiperparametru) w bieżącym runie.

**log_metric** — funkcja MLflow zapisująca wartość metryki (np. accuracy, F1) w bieżącym runie.

**log_artifact** — funkcja MLflow zapisująca plik (wykres, dane) jako artefakt bieżącego runu.

**log_model** — funkcja MLflow zapisująca model w standardowym formacie z opcjonalną sygnaturą wejścia/wyjścia.

### DVC

**DVC (Data Version Control)** — open-source system do wersjonowania danych, modeli i eksperymentów, zbudowany na bazie Git.

**DVC Pipelines** — reprodukowalne potoki przetwarzania danych definiowane w pliku dvc.yaml z automatycznym wykrywaniem zmian.

**DVC Experiments** — funkcjonalność DVC umożliwiająca szybkie testowanie parametrów bez commitowania każdego eksperymentu.

**dvc.yaml** — plik konfiguracyjny definiujący etapy pipeline'u, ich zależności, wyjścia i metryki.

**params.yaml** — plik z parametrami pipeline'u DVC; zmiana wartości powoduje ponowne wykonanie zależnych etapów.

**dvc repro** — komenda uruchamiająca pipeline DVC, wykonująca tylko etapy, których zależności się zmieniły.

**dvc dag** — komenda wyświetlająca graf zależności (DAG) między etapami pipeline'u.

**DAG (Directed Acyclic Graph)** — skierowany graf acykliczny opisujący zależności między etapami przetwarzania.

### Optuna

**create_study** — funkcja Optuna tworząca nowe studium (serię prób) z określonym kierunkiem optymalizacji.

**optimize** — metoda studium uruchamiająca zadaną liczbę prób optymalizacji funkcji celu.

**suggest_int** — metoda trial sugerująca wartość hiperparametru całkowitoliczbowego z zadanego zakresu.

**suggest_float** — metoda trial sugerująca wartość hiperparametru zmiennoprzecinkowego z zadanego zakresu.

**suggest_categorical** — metoda trial sugerująca wartość hiperparametru kategorycznego z listy opcji.

**MLflowCallback** — callback Optuna automatycznie logujący każdą próbę jako run w MLflow.

### BentoML

**BentoML** — framework do pakowania i wdrażania modeli ML jako serwisów API.

**Model Store** — lokalne repozytorium BentoML przechowujące zapisane modele z metadanymi.

**Service** — klasa BentoML definiująca endpointy API i logikę przetwarzania żądań.

**Bento** — pakiet zawierający model, serwis, zależności i konfigurację, gotowy do wdrożenia.

**Server** — serwer BentoML obsługujący żądania HTTP do endpointów serwisu.

**Endpoint** — punkt końcowy API udostępniający konkretną funkcjonalność (np. /predict, /predict_batch).

**bentoml serve** — komenda uruchamiająca serwis BentoML lokalnie jako serwer HTTP.

**bentoml build** — komenda budująca Bento (pakiet) z serwisu i modeli.

**bentoml containerize** — komenda tworząca obraz Docker z gotowego Bento.
