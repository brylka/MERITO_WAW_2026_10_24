# MODELE KLASYFIKACJI W MACHINE LEARNING

**Bartosz Bryniarski** · 2025

---

## Wprowadzenie do Klasyfikacji

Klasyfikacja to zadanie uczenia nadzorowanego, w którym algorytm uczy się przypisywać obserwacje do predefiniowanych kategorii (klas) na podstawie ich cech. Przykłady zastosowań:

- Rozpoznawanie chorób na podstawie objawów
- Filtrowanie spamu w emailach
- Rozpoznawanie obrazów (koty vs psy)
- Przewidywanie, czy klient dokona zakupu

**Podstawowe pojęcia:**

- **Cechy (features)** – zmienne wejściowe opisujące obserwację (np. waga, wzrost)
- **Etykiety (labels)** – kategorie, do których przypisujemy obserwacje (np. gatunek rośliny)
- **Zbiór treningowy** – dane używane do uczenia modelu
- **Zbiór testowy** – dane używane do oceny jakości modelu

---

## 1. DRZEWA DECYZYJNE (Decision Trees)

### Jak działają?

Drzewo decyzyjne to struktura przypominająca schemat blokowy, która podejmuje decyzje na podstawie serii pytań o cechy danych. Algorytm działa następująco:

1. **Wybór najlepszej cechy** – algorytm wybiera cechę, która najlepiej dzieli dane na grupy
2. **Podział danych** – dane dzielone są na podstawie wartości wybranej cechy
3. **Rekurencyjne powtarzanie** – proces powtarza się dla każdej podgrupy
4. **Zatrzymanie** – gdy wszystkie elementy w grupie należą do jednej klasy lub osiągnięto maksymalną głębokość

**Przykład decyzji:**

```
Czy długość płatka > 2.5cm?
├─ TAK → Czy szerokość płatka > 1.7cm?
│         ├─ TAK → Iris Virginica
│         └─ NIE → Iris Versicolor
└─ NIE → Iris Setosa
```

### Kluczowe parametry

**`criterion`** – miara jakości podziału:

- `gini` – indeks Giniego (domyślnie), mierzy „nieczystość” węzła
- `entropy` – entropia informacyjna, mierzy niepewność

**`max_depth`** – maksymalna głębokość drzewa:

- Kontroluje złożoność modelu
- Niższe wartości = prostsze drzewo = mniej przeuczenia
- Wyższe wartości = bardziej złożone drzewo = ryzyko overfittingu

**`min_samples_split`** – minimalna liczba próbek wymagana do podziału węzła:

- Wyższe wartości = bardziej konserwatywne dzielenie
- Pomaga zapobiegać przeuczeniu

### Zalety i wady

**Zalety:**

- Bardzo intuicyjne i łatwe do zrozumienia
- Można je wizualizować i wyjaśnić decyzje
- Nie wymagają normalizacji ani standaryzacji danych
- Radzą sobie z danymi numerycznymi i kategorycznymi
- Automatycznie wykrywają interakcje między cechami

**Wady:**

- Podatne na przeuczenie (overfitting)
- Niestabilne – małe zmiany w danych mogą całkowicie zmienić strukturę drzewa
- Mogą tworzyć złożone drzewa, które słabo generalizują
- Preferują cechy z wieloma wartościami

### Implementacja w Pythonie

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn import tree

# Wczytanie danych
iris = load_iris()
X, y = iris.data, iris.target

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Utworzenie i trenowanie modelu
dt_classifier = DecisionTreeClassifier(
    criterion='gini',  # lub 'entropy'
    max_depth=3,       # ograniczenie głębokości
    min_samples_split=2,
    random_state=42
)
dt_classifier.fit(X_train, y_train)

# Predykcja
y_pred = dt_classifier.predict(X_test)

# Ewaluacja
print(f"Dokładność: {accuracy_score(y_test, y_pred):.4f}")
print("\nRaport klasyfikacji:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Wizualizacja drzewa
plt.figure(figsize=(15, 10))
tree.plot_tree(dt_classifier, feature_names=iris.feature_names,
               class_names=iris.target_names, filled=True)
plt.show()
```

### Interpretacja wyników

**Dokładność (Accuracy)** – procent poprawnie sklasyfikowanych obserwacji

**Raport klasyfikacji zawiera:**

- **Precision** (precyzja) – ile z przewidzianych przypadków danej klasy było prawdziwie tą klasą
- **Recall** (czułość) – ile przypadków danej klasy udało się wykryć
- **F1-score** – średnia harmoniczna precision i recall
- **Support** – liczba rzeczywistych wystąpień klasy w zbiorze testowym

**Wizualizacja drzewa:**

- Każdy prostokąt to węzeł decyzyjny
- Kolory wskazują dominującą klasę w węźle
- Im intensywniejszy kolor, tym bardziej „czysta” jest klasa w węźle
- Liczby pokazują rozkład próbek z każdej klasy

---

## 2. K-NEAREST NEIGHBORS (KNN)

### Jak działa?

KNN to algorytm „leniwego uczenia” – nie buduje jawnego modelu, tylko zapamiętuje dane treningowe. Klasyfikacja nowej obserwacji przebiega następująco:

1. **Oblicz odległości** – mierzy się odległość nowej obserwacji do wszystkich punktów treningowych
2. **Znajdź k najbliższych sąsiadów** – wybiera się k punktów o najmniejszej odległości
3. **Głosowanie większościowe** – nowa obserwacja otrzymuje klasę, która jest najczęstsza wśród k sąsiadów

> **Intuicja:** „Powiedz mi, kim są twoi sąsiedzi, a powiem ci, kim jesteś”

### Kluczowe decyzje

**Wybór k:**

- **k = 1**: Każdy punkt klasyfikowany jak najbliższy sąsiad (ryzyko overfittingu, wrażliwy na szum)
- **k małe (3–5)**: Model bardziej złożony, lepiej dopasowany do szczegółów
- **k duże (>10)**: Model prostszy, bardziej wygładzony, bardziej odporny na szum
- **k nieparzyste**: Zapobiega remisowi w klasyfikacji binarnej

**Metryki odległości:**

- `euclidean` – odległość w linii prostej (najpopularniejsza)
- `manhattan` – suma różnic w każdym wymiarze (dobra dla przestrzeni miejskich)
- `minkowski` – uogólnienie euclidean i manhattan

**Wagi (`weights`):**

- `uniform` – wszyscy sąsiedzi mają równy głos
- `distance` – bliżsi sąsiedzi mają większy wpływ (lepsze dla niejednorodnych danych)

### Znaczenie standaryzacji

**Dlaczego standaryzacja jest kluczowa dla KNN?**

KNN opiera się na odległościach, więc cechy o większych wartościach numerycznych będą dominować w obliczeniach, nawet jeśli nie są ważniejsze.

**Przykład bez standaryzacji:**

- Cecha A: waga w gramach (50–100)
- Cecha B: wzrost w metrach (0.5–1.0)

Odległość będzie zdominowana przez wagę, bo ma większe wartości!

**StandardScaler:**

- Przekształca każdą cechę tak, aby miała średnią = 0 i odchylenie standardowe = 1
- Wszystkie cechy mają teraz porównywalny zakres
- `fit_transform()` na zbiorze treningowym – uczy się parametrów
- `transform()` na zbiorze testowym – stosuje te same parametry

### Implementacja w Pythonie

```python
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import load_iris
import numpy as np

# 1. ZAŁADOWANIE DANYCH (przykład z Iris dataset)
iris = load_iris()
X = iris.data
y = iris.target

# 2. PODZIAŁ NA ZBIÓR TRENINGOWY I TESTOWY
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. STANDARYZACJA DANYCH (ważne dla KNN!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. ZNALEZIENIE OPTYMALNEGO k
k_range = range(1, 31)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    scores.append(knn.score(X_test_scaled, y_test))

# 5. WIZUALIZACJA WYNIKÓW DLA RÓŻNYCH k
plt.figure(figsize=(10, 6))
plt.plot(k_range, scores, marker='o')
plt.xlabel('Wartość k')
plt.ylabel('Dokładność')
plt.title('Dokładność KNN dla różnych wartości k')
plt.grid(True)
plt.show()

# 6. TRENING Z OPTYMALNYM k
optimal_k = k_range[np.argmax(scores)]
print(f"Optymalne k: {optimal_k}")
print(f"Najwyższa dokładność: {max(scores):.4f}")

knn_classifier = KNeighborsClassifier(
    n_neighbors=optimal_k,
    weights='distance',  # lub 'uniform'
    metric='euclidean'   # lub 'manhattan', 'minkowski'
)
knn_classifier.fit(X_train_scaled, y_train)

# 7. PREDYKCJA I EWALUACJA
y_pred_knn = knn_classifier.predict(X_test_scaled)
print(f"\nDokładność KNN: {accuracy_score(y_test, y_pred_knn):.4f}")
print("\nMacierz pomyłek:")
print(confusion_matrix(y_test, y_pred_knn))
```

### Optymalizacja k

**Wykres dokładności vs k:**

- Niska wartość k (1–3): Wysoka zmienność, możliwe przeuczenie
- Optymalne k: Najwyższa dokładność na zbiorze testowym
- Wysoka wartość k: Model zbyt prosty, możliwe niedouczenie

**Strategia wyboru k:**

- Testuj zakres wartości (np. 1–30)
- Wybierz k z najwyższą dokładnością na zbiorze walidacyjnym
- Jeśli kilka wartości daje podobne wyniki, wybierz większe k (bardziej stabilne)

> **Uwaga:** w powyższym kodzie k jest wybierane na zbiorze testowym, co dla celów dydaktycznych jest w porządku, ale w praktyce zawyża ocenę modelu. Lepiej wybierać k na osobnym zbiorze walidacyjnym lub przez walidację krzyżową (np. `GridSearchCV`), a zbiór testowy zostawić do końcowej oceny.

### Zalety i wady

**Zalety:**

- Bardzo prosty i intuicyjny
- Nie wymaga trenowania (brak fazy uczenia)
- Naturalnie obsługuje klasyfikację wieloklasową
- Dobrze działa z nieliniowymi granicami decyzyjnymi

**Wady:**

- Bardzo wolny przy predykcji dla dużych zbiorów (musi obliczyć wszystkie odległości)
- Wymaga dużo pamięci (przechowuje wszystkie dane treningowe)
- Wrażliwy na skalę cech (konieczna standaryzacja)
- Wrażliwy na nieistotne cechy
- Słabo działa w wysokich wymiarach (curse of dimensionality)

---

## 3. NAIVE BAYES

### Jak działa?

Naive Bayes to klasyfikator probabilistyczny oparty na twierdzeniu Bayesa:

$$
P(\text{klasa} \mid \text{cechy}) = \frac{P(\text{cechy} \mid \text{klasa}) \times P(\text{klasa})}{P(\text{cechy})}
$$

**Założenie „naiwności”:** Algorytm zakłada, że wszystkie cechy są niezależne od siebie (dlatego „naiwny”). To założenie rzadko jest prawdziwe, ale mimo to algorytm często działa dobrze!

**Proces klasyfikacji:**

1. Dla każdej klasy oblicz prawdopodobieństwo P(klasa|cechy)
2. Wybierz klasę o najwyższym prawdopodobieństwie

### Warianty Naive Bayes

**GaussianNB** (użyty w kodzie):

- Dla cech ciągłych (numerycznych)
- Zakłada, że cechy mają rozkład normalny (Gaussa)
- Idealny dla danych takich jak Iris (wymiary płatków/działek)

**MultinomialNB:**

- Dla cech dyskretnych (zliczenia, częstości)
- Świetny dla klasyfikacji tekstu (liczba wystąpień słów)

**BernoulliNB:**

- Dla cech binarnych (0/1, tak/nie)
- Użyteczny dla wykrywania obecności/braku cech

### Implementacja w Pythonie

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import roc_curve, auc, accuracy_score, classification_report
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
import numpy as np

# 1. ZAŁADOWANIE DANYCH
iris = load_iris()
X = iris.data
y = iris.target

# 2. PODZIAŁ NA ZBIÓR TRENINGOWY I TESTOWY
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. GAUSSIAN NAIVE BAYES (dla cech ciągłych)
gnb_classifier = GaussianNB()
gnb_classifier.fit(X_train, y_train)

# 4. PREDYKCJA
y_pred_gnb = gnb_classifier.predict(X_test)
y_proba_gnb = gnb_classifier.predict_proba(X_test)

print(f"Dokładność Gaussian Naive Bayes: {accuracy_score(y_test, y_pred_gnb):.4f}")
print("\nRaport klasyfikacji:")
print(classification_report(y_test, y_pred_gnb, target_names=iris.target_names))

# 5. KRZYWA ROC DLA KLASYFIKACJI WIELOKLASOWEJ
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
n_classes = y_test_bin.shape[1]

plt.figure(figsize=(10, 8))
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba_gnb[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{iris.target_names[i]} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', label='Losowy klasyfikator')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Krzywa ROC - Naive Bayes')
plt.legend()
plt.grid(True)
plt.show()
```

### Krzywa ROC i AUC

**ROC (Receiver Operating Characteristic):**

- Pokazuje kompromis między czułością (True Positive Rate) a False Positive Rate
- Oś X: False Positive Rate (ilu zdrowych błędnie uznano za chorych)
- Oś Y: True Positive Rate (ilu chorych poprawnie zidentyfikowano)

**AUC (Area Under Curve):**

- Pole pod krzywą ROC
- AUC = 1.0: Idealny klasyfikator
- AUC = 0.5: Klasyfikator losowy (linia przerywana)
- AUC > 0.8: Dobry klasyfikator
- AUC > 0.9: Doskonały klasyfikator

**Klasyfikacja wieloklasowa:** Dla problemów z więcej niż dwiema klasami (jak Iris: 3 gatunki) tworzymy oddzielną krzywą ROC dla każdej klasy metodą „one-vs-rest”:

- Klasa A vs (B+C)
- Klasa B vs (A+C)
- Klasa C vs (A+B)

### Zalety i wady

**Zalety:**

- Bardzo szybki w treningu i predykcji
- Wymaga małej ilości danych treningowych
- Dobrze działa z wysokowymiarowymi danymi
- Doskonały dla klasyfikacji tekstu i analizy sentymentu
- Daje oszacowania prawdopodobieństw

**Wady:**

- Założenie niezależności cech jest często nierealistyczne
- Może dawać skrajne prawdopodobieństwa (bardzo bliskie 0 lub 1)
- Słabo działa, gdy zależności między cechami są ważne
- Wrażliwy na cechy niepowiązane z klasyfikacją

---

## 4. PORÓWNANIE MODELI

### Implementacja w Pythonie (KOD 4)

```python
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
import numpy as np

# 1. ZAŁADOWANIE DANYCH
iris = load_iris()
X = iris.data
y = iris.target

# 2. PODZIAŁ NA ZBIÓR TRENINGOWY I TESTOWY (do znalezienia optimal_k)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. INICJALIZACJA SCALERA
scaler = StandardScaler()

# 4. ZNALEZIENIE OPTYMALNEGO k DLA KNN
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

k_range = range(1, 31)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    scores.append(knn.score(X_test_scaled, y_test))

optimal_k = k_range[np.argmax(scores)]
print(f"Optymalne k: {optimal_k}\n")

# 5. PORÓWNANIE WSZYSTKICH TRZECH MODELI
models = {
    'Decision Tree': DecisionTreeClassifier(max_depth=3, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=optimal_k),
    'Naive Bayes': GaussianNB()
}

results = {}
print("Wyniki cross-validation:")
print("-" * 50)

for name, model in models.items():
    if name == 'KNN':
        # KNN wymaga standaryzacji
        X_scaled = scaler.fit_transform(X)
        scores = cross_val_score(model, X_scaled, y, cv=5)
    else:
        scores = cross_val_score(model, X, y, cv=5)

    results[name] = scores
    print(f"{name}: {scores.mean():.4f} (+/- {scores.std():.4f})")

# 6. WIZUALIZACJA PORÓWNANIA
plt.figure(figsize=(10, 6))
plt.boxplot(results.values(), tick_labels=results.keys())
plt.ylabel('Dokładność')
plt.title('Porównanie modeli klasyfikacji (5-fold CV)')
plt.grid(True)
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()
```

> **Uwaga (dobra praktyka):** w tym kodzie scaler jest dopasowywany do całego zbioru `X` przed walidacją krzyżową, więc informacja z foldów testowych „przecieka” do treningu. Na Iris różnica jest znikoma, ale poprawniej jest umieścić standaryzację w `Pipeline`, wtedy scaler uczy się osobno w każdym foldzie:
>
> ```python
> from sklearn.pipeline import make_pipeline
>
> knn_pipeline = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=optimal_k))
> scores = cross_val_score(knn_pipeline, X, y, cv=5)
> ```
>
> Parametr `tick_labels` w `plt.boxplot` działa od Matplotlib 3.9; w starszych wersjach użyj `labels`.

### Cross-Validation (Walidacja Krzyżowa)

**Dlaczego cross-validation?** Pojedynczy podział na train/test może być przypadkowy. Cross-validation daje bardziej wiarygodną ocenę.

**5-fold CV:**

- Dane dzielone na 5 równych części
- Model trenowany 5 razy, za każdym razem inna część jest testem
- Wynik to średnia z 5 prób

**Interpretacja wyników:**

- **Średnia** – ogólna wydajność modelu
- **Odchylenie standardowe** – stabilność modelu
  - Małe odchylenie = model stabilny
  - Duże odchylenie = model niestabilny, wrażliwy na dane

### Boxplot – Wizualizacja rozkładu wyników

**Elementy boxplota:**

- **Linia środkowa** – mediana (50. percentyl)
- **Pudełko** – zakres międzykwartylowy (25–75 percentyl)
- **Wąsy** – zasięg danych (bez outlierów)
- **Punkty** – wartości odstające (outliery)

**Co pokazuje dla modeli:**

- Wysokość pudełka: Jak bardzo wyniki się różnią
- Pozycja mediany: Typowa wydajność modelu
- Symetryczność: Czy model działa stabilnie

### Jak wybrać najlepszy model?

**Kryteria:**

- **Dokładność** – najwyższa średnia dokładność
- **Stabilność** – najmniejsze odchylenie standardowe
- **Interpretowalność** – czy można wyjaśnić decyzje?
- **Szybkość** – czas treningu i predykcji
- **Złożoność** – łatwość implementacji i utrzymania

**Dla zbioru Iris (typowe wyniki):**

- **Decision Tree**: ~95–97% – dobra równowaga, łatwa interpretacja
- **KNN**: ~93–96% – prosty, ale wymaga standaryzacji
- **Naive Bayes**: ~94–96% – bardzo szybki, dobre prawdopodobieństwa

### Praktyczne wskazówki

**Decision Trees – kiedy użyć:**

- Potrzebujesz interpretowalnego modelu
- Masz mieszane typy danych (numeryczne i kategoryczne)
- Chcesz wizualizować proces decyzyjny

**KNN – kiedy użyć:**

- Masz niewielki zbiór danych
- Granice decyzyjne są nieliniowe
- Nie potrzebujesz szybkich predykcji

**Naive Bayes – kiedy użyć:**

- Potrzebujesz bardzo szybkiego modelu
- Pracujesz z tekstem lub danymi o wysokiej wymiarowości
- Chcesz oszacowania prawdopodobieństw
- Masz małą ilość danych treningowych

---

## ZADANIA PRAKTYCZNE

### Zadanie 1: Eksperyment z parametrami Decision Tree

Zmień `max_depth` (2, 3, 5, 10, None) i porównaj:

- Jak zmienia się dokładność?
- Jak wygląda drzewo dla różnych głębokości?
- Który wariant jest najbardziej zrozumiały?

### Zadanie 2: Wpływ k w KNN

Uruchom kod dla różnych zakresów k (1–10, 1–50):

- Jak zmienia się optymalne k?
- Czy większy zakres daje lepsze wyniki?
- Co się dzieje dla k=1 vs k=20?

### Zadanie 3: Porównanie metryk odległości

W KNN zmień `metric` na `'manhattan'` i `'minkowski'`:

- Która metryka daje najlepsze wyniki?
- Czy optymalne k się zmienia?

### Zadanie 4: Analiza błędów

Dla każdego modelu przeanalizuj macierz pomyłek:

- Które klasy są najczęściej mylone?
- Dlaczego model popełnia te konkretne błędy?
- Co można poprawić?

### Zadanie 5: Własny dataset

Załaduj inny dataset (np. wine, breast_cancer) i:

- Przeprowadź pełną analizę wszystkimi trzema modelami
- Który model działa najlepiej?
- Czy wnioski są inne niż dla Iris?

---

## PODSUMOWANIE

| Model | Szybkość | Interpretowalność | Dokładność | Najlepsze dla |
| --- | --- | --- | --- | --- |
| Decision Tree | Szybka | Wysoka | Średnia–wysoka | Dane mieszane, wyjaśnialność |
| KNN | Wolna | Średnia | Średnia–wysoka | Małe zbiory, granice nieliniowe |
| Naive Bayes | Bardzo szybka | Średnia | Średnia | Tekst, wysokie wymiary |

> **Złota zasada:** Nie ma jednego „najlepszego” algorytmu. Zawsze testuj kilka podejść i wybieraj na podstawie wyników walidacji krzyżowej oraz wymagań projektu!
