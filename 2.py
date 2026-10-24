from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn import tree

# załadowanie danych
data = load_digits()
X, y = data.data, data.target

# print(data)

# podział na dane treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# stworzenie i trening modelu
dt_classifier = DecisionTreeClassifier(
                                       random_state=42)
dt_classifier.fit(X_train, y_train)

# predukcja - odpytanie modelu
y_pred = dt_classifier.predict(X_test)

# wyświetlenie dokładności
score = accuracy_score(y_test, y_pred)
print(f"Dokładność modelu: {score:.4f} ({score * 100:.2f}%)")

#print(classification_report(y_test, y_pred, target_names=data.target_names))

plt.figure(figsize=(30, 30))
tree.plot_tree(dt_classifier,
               filled=True)
plt.show()

# print(y_test)
# print(y_pred)




# print(iris.data)
# print(iris.target)
