from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)


dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

y_pred = dt_classifier.predict(X_test)

print(f"Dokładność modelu: {accuracy_score(y_test, y_pred):.4f} ({accuracy_score(y_test, y_pred) * 100:.2f}%)")



# print(y_test)
# print(y_pred)




# print(iris.data)
# print(iris.target)
