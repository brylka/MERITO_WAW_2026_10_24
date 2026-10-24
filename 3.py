from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# załadowanie danych
iris = load_iris()
X, y = iris.data, iris.target

# podział na dane treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# dodanie skalowania danych cech
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# print(X_test)
# print(X_test_scaled)

# wyszukiwanie optymalnego k
k_range = range(1,105)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    scores.append(knn.score(X_test_scaled, y_test))

# wykres
plt.figure(figsize=(15, 10))
plt.plot(k_range, scores, marker='o')
plt.title("Dokładność modelu dla różnych warości k")
plt.xlabel("Wartość k")
plt.ylabel("Dokładność")
plt.grid(True)
plt.show()

# znalezienie i wydrukowanie optymalnego k
optimal_k = k_range[np.argmax(scores)]
print(f"Optymalne k: {optimal_k}")
print(f"Największa dokłądność: {max(scores)}")

# trenowanie modelu z dodatkowymi parametrami
knn_clasifier = KNeighborsClassifier(
                        n_neighbors=optimal_k,
                        weights='distance', # 'uniform'
                        metric='euclidean', # 'manhattan', 'minkowski'
)
knn_clasifier.fit(X_train_scaled, y_train)

# predykcja na danych testowych przeskalowanych
y_pred = knn_clasifier.predict(X_test_scaled)

# wyświetlenie dokładności modelu
score = accuracy_score(y_test, y_pred)
print(f"Dokładność modelu przy {optimal_k} sąsiadach: {score:.4f} ({score * 100:.2f}%)")