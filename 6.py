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