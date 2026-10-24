from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# załadowanie danych
iris = load_iris()
X, y = iris.data, iris.target

# podział na dane treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

gnb_classifier = GaussianNB()
gnb_classifier.fit(X_train, y_train)

y_pred = gnb_classifier.predict(X_test)
y_proba = gnb_classifier.predict_proba(X_test)

print(f"Dokładność GNB: {accuracy_score(y_test, y_pred)}")
print(y_pred)
print(y_proba)