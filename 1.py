from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# załadowanie danych
iris = load_iris()
X, y = iris.data, iris.target

# podział na dane treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# stworzenie i trening modelu
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

# predukcja - odpytanie modelu
y_pred = dt_classifier.predict(X_test)

# wyświetlenie dokładności
score = accuracy_score(y_test, y_pred)
print(f"Dokładność modelu: {score:.4f} ({score * 100:.2f}%)")

print(classification_report(y_test, y_pred, target_names=iris.target_names))


# print(y_test)
# print(y_pred)




# print(iris.data)
# print(iris.target)
