from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np

# załadowanie danych
iris = load_iris()
X, y = iris.data, iris.target

# podział na dane treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# print(X_test)
# print(X_test_scaled)

k_range = range(1,51)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    scores.append(knn.score(X_test_scaled, y_test))

# print(scores)

optimal_k = k_range[np.argmax(scores)]
print(f"Optymalne k: {optimal_k}")
print(f"Największa dokłądność: {max(scores)}")




    # y_pred = knn.predict(X_test_scaled)
    #
    # score = accuracy_score(y_test, y_pred)
    # print(f"Dokładność modelu przy {k} sąsiadach: {score:.4f} ({score * 100:.2f}%)")