from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

X_blobs, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)

# print(X_blobs)
# print(y_true)

plt.figure(figsize=(15,10))

plt.scatter(X_blobs[:, 0], X_blobs[:, 1], c=y_true, alpha=0.6)
plt.show()