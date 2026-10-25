from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_digits
import joblib

# załadowanie danych
print("Wczytanie danych.")
data = load_digits()
X, y = data.data, data.target

# stworzenie i trening modelu
print("Trenowanie modelu.")
model = DecisionTreeClassifier()
model.fit(X, y)

# zapisanie pliku modelu
joblib.dump(model, "model.joblib")
print("Zapisano plik modelu.")
