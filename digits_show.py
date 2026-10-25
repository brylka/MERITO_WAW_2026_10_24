from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

index = 9
print("Wczytanie danych.")
data = load_digits()
X, y = data.data[index], data.target[index]

print(X, y)

plt.imshow(X.reshape(8,8), cmap="gray")
plt.show()