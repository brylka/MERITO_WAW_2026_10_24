from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


X_blobs, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)

inertias = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_blobs)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(10, 10))
plt.plot(K_range, inertias, 'o-')
plt.grid(True)
plt.show()



# print(X_blobs)
# print(y_true)

# plt.figure(figsize=(15,10))
# plt.scatter(X_blobs[:, 0], X_blobs[:, 1], c=y_true, alpha=0.6)
# plt.show()