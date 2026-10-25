# KLASTERYZACJA W MACHINE LEARNINGU

**Bartosz Bryniarski** · 2025

---

## Wprowadzenie do Klasteryzacji

Klasteryzacja to zadanie **uczenia nienadzorowanego** (unsupervised learning), w którym algorytm grupuje dane w klastry bez wcześniejszego poznania etykiet. W przeciwieństwie do klasyfikacji nie mamy z góry określonych kategorii – algorytm sam odkrywa naturalne grupy w danych.

**Podstawowa różnica:**

- **Klasyfikacja** (supervised): Mamy etykiety → uczymy model → przewidujemy etykiety nowych danych
- **Klasteryzacja** (unsupervised): Nie mamy etykiet → szukamy naturalnych grup → odkrywamy strukturę danych

**Przykłady zastosowań klasteryzacji:**

- Segmentacja klientów w marketingu (grupy o podobnych zachowaniach zakupowych)
- Kompresja obrazów (grupowanie podobnych kolorów)
- Wykrywanie anomalii (punkty nienależące do żadnego klastra)
- Grupowanie dokumentów według tematyki
- Analiza danych biologicznych (grupowanie genów o podobnej ekspresji)
- Organizacja bibliotek zdjęć (grupowanie podobnych obrazów)

**Kluczowe pojęcia:**

- **Klaster** – grupa podobnych do siebie obiektów
- **Centroid** – punkt reprezentujący środek klastra
- **Odległość** – miara podobieństwa między punktami
- **Outlier** – punkt odstający, nienależący do żadnego klastra

---

## 1. K-MEANS

### Jak działa K-means?

K-means to najpopularniejszy algorytm klasteryzacji. Dzieli dane na **k** klastrów, gdzie k jest określone przez użytkownika. Każdy klaster reprezentowany jest przez centroid – punkt będący średnią wszystkich punktów w klastrze.

**Algorytm krok po kroku:**

1. **Inicjalizacja**: Losowo wybierz k punktów jako początkowe centroidy
2. **Przypisanie**: Przypisz każdy punkt danych do najbliższego centroidu (powstają klastry)
3. **Aktualizacja**: Przelicz centroidy jako średnią arytmetyczną wszystkich punktów w każdym klastrze
4. **Iteracja**: Powtarzaj kroki 2–3, aż:
   - centroidy przestaną się zmieniać (zbieżność),
   - osiągnięto maksymalną liczbę iteracji,
   - zmiany są poniżej określonego progu.

**Intuicja:** Wyobraź sobie, że masz 100 punktów na płaszczyźnie i chcesz je podzielić na 3 grupy. K-means najpierw losowo „rzuca” 3 znaczniki (centroidy), a następnie:

- Każdy punkt „przyciąga się” do najbliższego znacznika
- Znaczniki przesuwają się do środka swoich grup
- Proces powtarza się, aż wszystko się ustabilizuje

### Kluczowe parametry

**`n_clusters` (k)** – liczba klastrów:

- Najbardziej krytyczny parametr
- Musi być określony z góry
- Użyj metody łokcia lub współczynnika sylwetkowego do wyboru optymalnego k

**`init`** – metoda inicjalizacji centroidów:

- `'k-means++'` (domyślnie, zalecane) – inteligentna inicjalizacja, która wybiera centroidy tak, aby były od siebie oddalone
- `'random'` – losowy wybór k punktów jako centroidów
- K-means++ znacznie poprawia zbieżność i jakość wyników

**`n_init`** – liczba uruchomień algorytmu:

- Domyślnie `'auto'` (od scikit-learn 1.4): 1 uruchomienie dla `init='k-means++'`, 10 dla `init='random'`; we wcześniejszych wersjach domyślnie 10
- Algorytm uruchamiany jest `n_init` razy z różnymi inicjalizacjami
- Wybierany jest najlepszy wynik (najniższa inercja)
- Zapobiega utknięciu w lokalnym minimum

**`max_iter`** – maksymalna liczba iteracji:

- Domyślnie 300
- Limit iteracji dla jednego uruchomienia algorytmu

**`random_state`** – ziarno losowości:

- Zapewnia powtarzalność wyników

### Metoda łokcia (Elbow Method)

**Po co?** Pomaga wybrać optymalną liczbę klastrów k.

**Jak to działa:**

1. Trenuj K-means dla różnych wartości k (np. 2–10)
2. Dla każdego k oblicz **inercję** (suma kwadratów odległości punktów do ich centroidów)
3. Narysuj wykres k vs inercja
4. Szukaj „łokcia” – punktu, gdzie inercja przestaje szybko spadać

**Interpretacja:**

- Mała wartość k → duża inercja (dane słabo zgrupowane)
- Duża wartość k → mała inercja (każdy punkt może być swoim klastrem)
- **Łokieć** to kompromis: k wystarczająco duże, ale nie za duże

### Współczynnik sylwetkowy (Silhouette Score)

**Po co?** Alternatywna metoda wyboru k, która ocenia jakość klasteryzacji.

**Jak to działa:** Dla każdego punktu oblicza się:

- **a** – średnia odległość do innych punktów w tym samym klastrze (spójność)
- **b** – średnia odległość do punktów w najbliższym innym klastrze (separacja)

$$
s = \frac{b - a}{\max(a, b)}
$$

**Wartości:**

- **+1**: Punkt idealnie dopasowany do swojego klastra
- **0**: Punkt na granicy między klastrami
- **−1**: Punkt prawdopodobnie przypisany do złego klastra

**Średnia dla wszystkich punktów:**

- Bliżej 1 = lepsze klastry (zwarte i dobrze oddzielone)
- Bliżej 0 = klastry się nakładają
- Wartości ujemne = źle dobrana liczba klastrów

### Davies-Bouldin Index

**Po co?** Kolejna metryka jakości klasteryzacji.

**Jak to działa:**

- Mierzy stosunek: (rozproszenie w klastrach) / (odległość między klastrami)
- Niższe wartości = lepsze klastry

**Interpretacja:**

- **Wartość 0**: Perfekcyjna klasteryzacja
- **Im niższa, tym lepiej**: Klastry zwarte i oddalone od siebie
- Wartości typowo w zakresie 0.5–2.5

### Implementacja w Pythonie (Metoda łokcia i optymalizacja)

```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt

# Generowanie danych przykładowych
X_blobs, y_true = make_blobs(n_samples=300, centers=4,
                             cluster_std=0.60, random_state=42)

# Metoda łokcia (Elbow Method) - wybór optymalnej liczby klastrów
inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_blobs)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_blobs, kmeans.labels_))

# Wizualizacja metody łokcia
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(K_range, inertias, 'bo-')
ax1.set_xlabel('Liczba klastrów (k)')
ax1.set_ylabel('Inercja (suma kwadratów odległości)')
ax1.set_title('Metoda łokcia')
ax1.grid(True)

ax2.plot(K_range, silhouette_scores, 'ro-')
ax2.set_xlabel('Liczba klastrów (k)')
ax2.set_ylabel('Współczynnik sylwetkowy')
ax2.set_title('Współczynnik sylwetkowy dla różnych k')
ax2.grid(True)

plt.tight_layout()
plt.show()

# K-means z optymalną liczbą klastrów
optimal_k = 4
kmeans = KMeans(
    n_clusters=optimal_k,
    init='k-means++',  # inteligentna inicjalizacja
    n_init=10,
    max_iter=300,
    random_state=42
)
y_kmeans = kmeans.fit_predict(X_blobs)

# Wizualizacja wyników
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_blobs[:, 0], X_blobs[:, 1], c=y_true, cmap='viridis', alpha=0.6)
plt.title('Prawdziwe etykiety')
plt.xlabel('Cecha 1')
plt.ylabel('Cecha 2')

plt.subplot(1, 2, 2)
plt.scatter(X_blobs[:, 0], X_blobs[:, 1], c=y_kmeans, cmap='viridis', alpha=0.6)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=300, c='red', marker='X', edgecolors='black', linewidths=2, label='Centroidy')
plt.title('Klasteryzacja K-means')
plt.xlabel('Cecha 1')
plt.ylabel('Cecha 2')
plt.legend()

plt.tight_layout()
plt.show()

# Metryki ewaluacji
print(f"Współczynnik sylwetkowy: {silhouette_score(X_blobs, y_kmeans):.4f}")
print(f"Davies-Bouldin Index: {davies_bouldin_score(X_blobs, y_kmeans):.4f}")
print(f"Inercja: {kmeans.inertia_:.2f}")
```

**Co robi ten kod:**

- Tworzy syntetyczne dane z 4 klastrami (gaussowskimi „bąbelkami”)
- Testuje K-means dla k od 2 do 10
- Rysuje wykresy inercji i sylwetki – pomagają wybrać optymalne k
- Trenuje model z optymalnym k=4
- Wizualizuje prawdziwe etykiety vs wyniki K-means
- Wyświetla metryki jakości

### Zalety i wady K-means

**Zalety:**

- Bardzo szybki i skalowalny – działa dobrze z dużymi zbiorami danych
- Prosty w implementacji i zrozumieniu
- Gwarantowana zbieżność (zatrzyma się w skończonej liczbie iteracji)
- Działa dobrze z „kulowymi” klastrami (okrągłymi, o podobnej wielkości)
- Mało pamięci – przechowuje tylko centroidy

**Wady:**

- **Musi znać k z góry** – nie odkrywa liczby klastrów automatycznie
- Wrażliwy na inicjalizację – może utknąć w lokalnym minimum (dlatego `n_init` > 1)
- Zakłada klastry kulowe o podobnej wielkości
- **Nie radzi sobie z:**
  - klastrami o nieregularnych kształtach (księżyce, pierścienie),
  - klastrami o różnej gęstości,
  - outlierami – każdy punkt musi być w jakimś klastrze.
- Wrażliwy na skalę cech – wymaga standaryzacji

### Kiedy używać K-means?

**Użyj K-means, gdy:**

- Wiesz lub możesz oszacować liczbę klastrów
- Klastry mają kulowy kształt i podobną wielkość
- Potrzebujesz szybkiego algorytmu dla dużych danych
- Dane są w miarę równomiernie rozłożone

**Unikaj K-means, gdy:**

- Klastry mają nieregularne kształty (użyj DBSCAN)
- Klastry mają bardzo różną gęstość
- Masz dużo outlierów
- Nie wiesz, ile jest klastrów

---

## 2. DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

### Jak działa DBSCAN?

DBSCAN to algorytm oparty na **gęstości** – identyfikuje klastry jako obszary o wysokiej gęstości punktów, oddzielone obszarami o niskiej gęstości. W przeciwieństwie do K-means DBSCAN:

- Nie wymaga określania liczby klastrów z góry
- Automatycznie wykrywa outliery
- Radzi sobie z klastrami o nieregularnych kształtach

**Rodzaje punktów w DBSCAN:**

- **Core point** (punkt rdzeniowy):
  - ma co najmniej `min_samples` punktów w promieniu `eps`,
  - stanowi „rdzeń” klastra.
- **Border point** (punkt brzegowy):
  - ma mniej niż `min_samples` sąsiadów,
  - ale jest w zasięgu `eps` od core pointa,
  - należy do klastra, ale na jego obrzeżu.
- **Noise point** (outlier):
  - nie jest core pointem,
  - nie jest w zasięgu żadnego core pointa,
  - oznaczany etykietą −1.

**Algorytm krok po kroku:**

1. Wybierz losowy punkt, który nie był jeszcze odwiedzony
2. Znajdź wszystkie punkty w promieniu `eps` od niego
3. Jeśli jest ich co najmniej `min_samples`:
   - utwórz nowy klaster,
   - rekurencyjnie dodawaj wszystkie osiągalne punkty do klastra.
4. Jeśli nie ma wystarczająco sąsiadów:
   - oznacz punkt jako noise (może być później przypisany jako border point).
5. Powtarzaj dla wszystkich nieodwiedzonych punktów

### Kluczowe parametry

**`eps` (epsilon)** – promień sąsiedztwa:

- Maksymalna odległość między dwoma punktami, aby uznać je za sąsiadów
- **Najbardziej krytyczny parametr**
- Za małe eps → wszystko to noise
- Za duże eps → wszystko to jeden klaster
- Metoda wyboru: wykres k-distance (opisany poniżej)

**`min_samples`** – minimalna liczba sąsiadów:

- Minimalna liczba punktów w promieniu `eps` (wliczając sam punkt), aby utworzyć core point
- Typowo: 2 × liczba wymiarów (dla 2D → `min_samples=4` lub 5)
- Większe wartości → bardziej konserwatywne klastry, więcej noise
- Mniejsze wartości → więcej małych klastrów

**`metric`** – metryka odległości:

- `'euclidean'` (domyślnie) – standardowa odległość w linii prostej
- `'manhattan'` – suma różnic bezwzględnych
- `'cosine'` – dla danych tekstowych
- Inna, niestandardowa metryka

### Wybór parametru eps – metoda k-distance

**Problem:** Jak wybrać odpowiednie eps?

**Rozwiązanie – wykres k-distance:**

1. Dla każdego punktu znajdź odległość do k-tego najbliższego sąsiada (gdzie k = `min_samples`)
2. Posortuj te odległości rosnąco
3. Narysuj wykres
4. Szukaj „łokcia” – punktu, gdzie krzywa gwałtownie rośnie

**Interpretacja:**

- **Przed łokciem**: Punkty gęsto upakowane (wewnątrz klastrów)
- **Po łokciu**: Punkty rzadkie (między klastrami lub noise)
- **Punkt łokcia** = optymalne eps

### Implementacja w Pythonie (DBSCAN podstawowy)

```python
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt

# Generowanie danych o nieregularnym kształcie
X_moons, y_moons = make_moons(n_samples=300, noise=0.05, random_state=42)

# Standaryzacja (ważna dla DBSCAN!)
scaler = StandardScaler()
X_moons_scaled = scaler.fit_transform(X_moons)

# Znalezienie optymalnego eps - metoda k-distance
neighbors = NearestNeighbors(n_neighbors=5)
neighbors.fit(X_moons_scaled)
distances, indices = neighbors.kneighbors(X_moons_scaled)

# Sortowanie odległości
distances = np.sort(distances[:, -1], axis=0)

plt.figure(figsize=(10, 6))
plt.plot(distances)
plt.xlabel('Punkty posortowane według odległości')
plt.ylabel('5-ta najbliższa odległość')
plt.title('Wykres k-distance dla wyboru eps')
plt.grid(True)
plt.show()

# DBSCAN z wybranymi parametrami
dbscan = DBSCAN(
    eps=0.3,           # promień sąsiedztwa
    min_samples=5,     # minimalna liczba punktów
    metric='euclidean'
)
y_dbscan = dbscan.fit_predict(X_moons_scaled)

# Liczba klastrów i outlierów
n_clusters = len(set(y_dbscan)) - (1 if -1 in y_dbscan else 0)
n_noise = list(y_dbscan).count(-1)

print(f"Liczba klastrów: {n_clusters}")
print(f"Liczba outlierów: {n_noise}")

# Wizualizacja wyników
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Prawdziwe etykiety
axes[0].scatter(X_moons[:, 0], X_moons[:, 1], c=y_moons, cmap='viridis', alpha=0.6)
axes[0].set_title('Prawdziwe etykiety')
axes[0].set_xlabel('Cecha 1')
axes[0].set_ylabel('Cecha 2')

# K-means (dla porównania)
kmeans_moons = KMeans(n_clusters=2, random_state=42)
y_kmeans_moons = kmeans_moons.fit_predict(X_moons)

axes[1].scatter(X_moons[:, 0], X_moons[:, 1], c=y_kmeans_moons, cmap='viridis', alpha=0.6)
axes[1].scatter(kmeans_moons.cluster_centers_[:, 0], kmeans_moons.cluster_centers_[:, 1],
                s=300, c='red', marker='X', edgecolors='black', linewidths=2)
axes[1].set_title('K-means (nie radzi sobie)')
axes[1].set_xlabel('Cecha 1')
axes[1].set_ylabel('Cecha 2')

# DBSCAN
axes[2].scatter(X_moons[:, 0], X_moons[:, 1], c=y_dbscan, cmap='viridis', alpha=0.6)

# Oznacz outliery tylko jeśli istnieją
outliers = X_moons[y_dbscan == -1]
if len(outliers) > 0:
    axes[2].scatter(outliers[:, 0], outliers[:, 1], c='red', marker='x',
                    s=100, linewidths=2, label='Outliers')
    axes[2].legend()  # Legenda tylko gdy są outliery

axes[2].set_title(f'DBSCAN (radzi sobie dobrze) - {n_noise} outliers')
axes[2].set_xlabel('Cecha 1')
axes[2].set_ylabel('Cecha 2')

plt.tight_layout()
plt.show()
```

**Co robi ten kod:**

- Tworzy dane w kształcie dwóch „księżyców” – niemożliwe dla K-means!
- Standaryzuje dane (DBSCAN wymaga porównywalnych skal)
- Używa metody k-distance do znalezienia optymalnego eps
- Trenuje DBSCAN i identyfikuje klastry + outliery
- Porównuje wyniki: prawda vs K-means vs DBSCAN
- Pokazuje przewagę DBSCAN nad K-means dla nieregularnych kształtów

### Problem z różną gęstością

```python
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Dane z klastrami o różnej gęstości
X_varied, y_varied = make_blobs(n_samples=[100, 50, 150],
                                centers=[[0, 0], [3, 3], [6, 1]],
                                cluster_std=[0.5, 1.5, 0.3],
                                random_state=42)

# DBSCAN na danych o różnej gęstości
dbscan_varied = DBSCAN(eps=0.5, min_samples=5)
y_dbscan_varied = dbscan_varied.fit_predict(X_varied)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_varied[:, 0], X_varied[:, 1], c=y_varied, cmap='viridis', alpha=0.6)
plt.title('Prawdziwe klastry (różna gęstość)')
plt.xlabel('Cecha 1')
plt.ylabel('Cecha 2')

plt.subplot(1, 2, 2)
plt.scatter(X_varied[:, 0], X_varied[:, 1], c=y_dbscan_varied, cmap='viridis', alpha=0.6)
outliers_varied = X_varied[y_dbscan_varied == -1]
if len(outliers_varied) > 0:
    plt.scatter(outliers_varied[:, 0], outliers_varied[:, 1], c='red',
                marker='x', s=100, linewidths=2, label='Outliers')
    plt.legend()  # Legenda tylko gdy są outliery
plt.title('DBSCAN (problemy z różną gęstością)')
plt.xlabel('Cecha 1')
plt.ylabel('Cecha 2')

plt.tight_layout()
plt.show()
```

**Co pokazuje ten kod:**

- Tworzy 3 klastry o różnym odchyleniu standardowym (różna gęstość)
- DBSCAN ma problem: jedno eps nie pasuje do wszystkich
- Zbyt małe eps → rzadkie klastry rozpadają się na noise
- Zbyt duże eps → klastry się łączą
- To główna wada DBSCAN!

### Zalety i wady DBSCAN

**Zalety:**

- **Nie wymaga określenia liczby klastrów k** – odkrywa ją automatycznie
- Radzi sobie z klastrami o dowolnym kształcie (księżyce, spirale, pierścienie)
- **Automatycznie wykrywa outliery** – punkty odstające oznaczone jako −1
- Brak losowej inicjalizacji – wynik jest deterministyczny (jedynie przypisanie punktów brzegowych może zależeć od kolejności danych)
- Tylko 2 parametry do dostrojenia (`eps`, `min_samples`)

**Wady:**

- **Wrażliwy na parametry `eps` i `min_samples`** – trudny dobór
- Problemy z klastrami o **różnej gęstości**
- Nie radzi sobie z danymi wysokowymiarowymi (curse of dimensionality)
- Wolniejszy niż K-means dla bardzo dużych zbiorów
- Brak centroidów – nie można łatwo przewidzieć klastra dla nowego punktu (brak metody `predict`)
- Wymaga standaryzacji danych

### Kiedy używać DBSCAN?

**Użyj DBSCAN, gdy:**

- Nie wiesz, ile jest klastrów
- Klastry mają nieregularne kształty
- Chcesz automatycznie wykrywać outliery
- Klastry mają podobną gęstość
- Masz dane przestrzenne (geograficzne)

**Unikaj DBSCAN, gdy:**

- Klastry mają bardzo różną gęstość (użyj HDBSCAN – w scikit-learn od wersji 1.3 jako `sklearn.cluster.HDBSCAN`)
- Dane są wysokowymiarowe (>20 wymiarów)
- Potrzebujesz bardzo szybkiego algorytmu dla milionów punktów
- Chcesz przewidywać klaster dla nowych punktów

---

## 3. PORÓWNANIE K-MEANS VS DBSCAN

### Kompleksowe porównanie na różnych danych

```python
from sklearn.datasets import make_circles, make_blobs, make_moons
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

datasets = [
    make_blobs(n_samples=300, centers=3, cluster_std=0.6, random_state=42),
    make_moons(n_samples=300, noise=0.05, random_state=42),
    make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=42)
]
dataset_names = ['Kulowe klastry', 'Klastry księżycowe', 'Klastry okrągłe']

fig, axes = plt.subplots(3, 3, figsize=(15, 15))

for i, (X, y) in enumerate(datasets):
    # Standaryzacja
    X_scaled = StandardScaler().fit_transform(X)

    # Oryginalne dane
    axes[i, 0].scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.6)
    axes[i, 0].set_title(f'{dataset_names[i]} - Oryginał')
    axes[i, 0].set_ylabel('Cecha 2')

    # K-means (liczba klastrów = prawdziwa liczba grup w danym zbiorze)
    kmeans = KMeans(n_clusters=len(set(y)), random_state=42)
    y_kmeans = kmeans.fit_predict(X)
    axes[i, 1].scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis', alpha=0.6)
    axes[i, 1].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                       s=200, c='red', marker='X', edgecolors='black', linewidths=2)
    axes[i, 1].set_title(f'K-means (Silhouette: {silhouette_score(X, y_kmeans):.3f})')

    # DBSCAN
    dbscan = DBSCAN(eps=0.3, min_samples=5)
    y_dbscan = dbscan.fit_predict(X_scaled)
    axes[i, 2].scatter(X[:, 0], X[:, 1], c=y_dbscan, cmap='viridis', alpha=0.6)
    outliers = X[y_dbscan == -1]
    if len(outliers) > 0:
        axes[i, 2].scatter(outliers[:, 0], outliers[:, 1], c='red',
                           marker='x', s=100, linewidths=2)

    # Oblicz silhouette tylko dla punktów niebędących outlierami
    mask = y_dbscan != -1
    if mask.sum() > 0 and len(set(y_dbscan[mask])) > 1:
        sil_score = silhouette_score(X[mask], y_dbscan[mask])
    else:
        sil_score = 0
    axes[i, 2].set_title(f'DBSCAN (Silhouette: {sil_score:.3f})')

for ax in axes[-1]:
    ax.set_xlabel('Cecha 1')

plt.tight_layout()
plt.show()
```

**Co pokazuje ten kod:**

- Testuje oba algorytmy na 3 różnych typach danych
- **Kulowe klastry**: K-means wygrywa (prostsze i szybsze)
- **Księżyce**: DBSCAN zdecydowanie lepszy
- **Okręgi koncentryczne**: DBSCAN radzi sobie, K-means nie
- Pokazuje współczynnik sylwetkowy dla porównania jakości

> **Uwaga:** współczynnik sylwetkowy premiuje zwarte, „kuliste” klastry. Dla księżyców i okręgów może więc wyjść **wyższy dla K-means**, mimo że wizualnie poprawny podział daje DBSCAN. To dobra ilustracja, że metryk wewnętrznych nie należy interpretować w oderwaniu od wizualizacji.

### Tabela porównawcza

| Cecha | K-means | DBSCAN |
| --- | --- | --- |
| **Liczba klastrów** | Trzeba znać z góry | Wykrywa automatycznie |
| **Kształt klastrów** | Kulowy | Dowolny |
| **Outliery** | Zawsze przypisuje do klastra | Automatycznie wykrywa |
| **Szybkość** | Bardzo szybki | Wolniejszy |
| **Parametry** | k, init, n_init | eps, min_samples |
| **Standaryzacja** | Zalecana | Konieczna |
| **Różna gęstość** | Nie radzi sobie | Problemy |
| **Wysokie wymiary** | Średnio | Słabo |
| **Przewidywanie nowych punktów** | Łatwe | Trudne |
| **Interpretowalność** | Wysoka (centroidy) | Średnia |

### Kiedy co wybrać?

**Wybierz K-means, gdy:**

- Masz kulowe klastry o podobnej wielkości
- Wiesz lub możesz oszacować liczbę klastrów
- Potrzebujesz szybkości i skalowalności
- Chcesz prostej interpretacji (centroidy)
- Przykład: segmentacja klientów, kompresja kolorów

**Wybierz DBSCAN, gdy:**

- Nie wiesz, ile jest klastrów
- Klastry mają nieregularne kształty
- Masz outliery do wykrycia
- Masz dane przestrzenne lub geograficzne
- Przykład: analiza anomalii, grupowanie lokalizacji GPS

---

## ZADANIA PRAKTYCZNE

### Zadanie 1: Eksperyment z liczbą klastrów w K-means

Użyj datasetu Iris i:

- Przetestuj K-means dla k=2, 3, 4, 5
- Narysuj metodę łokcia i współczynnik sylwetkowy
- Które k jest optymalne? Dlaczego?
- Porównaj z prawdziwą liczbą gatunków (3)

### Zadanie 2: Wpływ inicjalizacji w K-means

Uruchom K-means z `init='random'` i `n_init=1` kilka razy (z różnymi `random_state`):

- Czy dostajesz za każdym razem takie same wyniki?
- Porównaj z `init='k-means++'` i `n_init=10`
- Dlaczego k-means++ jest lepszy?

### Zadanie 3: Dobór parametrów DBSCAN

Na danych `make_moons`:

- Przetestuj różne wartości eps (0.1, 0.3, 0.5, 1.0)
- Przetestuj różne `min_samples` (3, 5, 10)
- Jak parametry wpływają na liczbę klastrów i outlierów?
- Znajdź optymalne parametry, używając wykresu k-distance

### Zadanie 4: Porównanie na własnych danych

Wygeneruj dane, używając `make_circles`:

```python
from sklearn.datasets import make_circles

X, y = make_circles(n_samples=500, noise=0.05, factor=0.5, random_state=42)
```

- Zastosuj K-means i DBSCAN
- Który algorytm działa lepiej? Dlaczego?
- Co się stanie, gdy zmienisz parametr `factor`?

### Zadanie 5: Analiza outlierów

Użyj DBSCAN na danych z ręcznie dodanymi outlierami:

```python
import numpy as np
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=300, centers=3, random_state=42)

# Dodaj outliery
outliers = np.random.uniform(low=-10, high=10, size=(20, 2))
X = np.vstack([X, outliers])
```

- Ile outlierów wykrył DBSCAN?
- Czy są to rzeczywiście punkty odstające?
- Co się stanie, gdy zwiększysz eps?

### Zadanie 6: Klasteryzacja rzeczywistych danych

Załaduj dataset Wine lub Breast Cancer:

- Usuń etykiety i zastosuj klasteryzację
- Porównaj wyniki z prawdziwymi etykietami (np. za pomocą `adjusted_rand_score` – patrz niżej)
- Która metoda lepiej odkrywa naturalne grupy?

---

## METRYKI EWALUACJI KLASTERYZACJI

Ponieważ klasteryzacja jest uczeniem nienadzorowanym, zwykle nie mamy „prawdziwych” etykiet do porównania. Używamy wewnętrznych metryk jakości:

**1. Inercja (Within-Cluster Sum of Squares)**

- Suma kwadratów odległości punktów do ich centroidów
- Niższa = lepiej
- Używana w metodzie łokcia
- Maleje wraz ze wzrostem k

**2. Współczynnik sylwetkowy (Silhouette Score)**

- Zakres: [−1, 1]
- Mierzy, jak dobrze punkt pasuje do swojego klastra vs innych
- Wyższy = lepiej
- Idealny do wyboru k

**3. Davies-Bouldin Index**

- Zakres: [0, ∞)
- Niższy = lepiej
- Stosunek rozproszenia wewnątrz klastrów do separacji między klastrami
- 0 = perfekcyjna klasteryzacja

**4. Calinski-Harabasz Index (Variance Ratio)**

- Zakres: [0, ∞)
- Wyższy = lepiej
- Stosunek rozproszenia między klastrami do rozproszenia wewnątrz klastrów
- Szybszy do obliczenia niż sylwetka

**Jak używać:**

```python
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

# X - dane, labels - etykiety klastrów zwrócone przez algorytm
print(f"Silhouette: {silhouette_score(X, labels):.3f}")
print(f"Davies-Bouldin: {davies_bouldin_score(X, labels):.3f}")
print(f"Calinski-Harabasz: {calinski_harabasz_score(X, labels):.3f}")
```

> **Metryki zewnętrzne:** jeśli prawdziwe etykiety są dostępne (jak w zadaniach 1 i 6), można porównać z nimi wynik klasteryzacji. Najczęściej używa się **Adjusted Rand Index** (`adjusted_rand_score`, zakres do 1, ~0 = przypadkowy podział) oraz **Normalized Mutual Information** (`normalized_mutual_info_score`, zakres [0, 1]). Obie metryki nie zależą od tego, jakie numery algorytm nadał klastrom.
>
> ```python
> from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
>
> print(f"ARI: {adjusted_rand_score(y_true, labels):.3f}")
> print(f"NMI: {normalized_mutual_info_score(y_true, labels):.3f}")
> ```

---

## PODSUMOWANIE

**K-means:**

- Prosty, szybki, popularny
- Wymaga znajomości k
- Klastry kulowe
- Świetny punkt startowy

**DBSCAN:**

- Automatycznie wykrywa liczbę klastrów
- Dowolne kształty klastrów
- Wykrywa outliery
- Wrażliwy na parametry

> **Złota zasada:** Zawsze zaczynaj od wizualizacji danych! Kształt i struktura danych podpowiedzą, który algorytm wybrać. Testuj oba i porównuj wyniki, używając metryk i wizualizacji.

**Praktyczny workflow:**

1. Zwizualizuj dane (scatter plot)
2. Standaryzuj cechy
3. Jeśli klastry są kulowe → K-means (metoda łokcia dla k)
4. Jeśli są nieregularne → DBSCAN (k-distance dla eps)
5. Oblicz metryki jakości
6. Zwaliduj wyniki wizualnie
7. Jeśli wynik nie jest zadowalający, dostosuj parametry i powtórz
