from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, make_moons
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

#X_blobs, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)
X_blobs, y_true = make_moons(n_samples=300, noise=0.05, random_state=42)


inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_blobs)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_blobs, kmeans.labels_))

# plt.figure(figsize=(10, 8))
# plt.plot(K_range, inertias, 'o-')
# plt.grid(True)
# plt.show()
#
# plt.figure(figsize=(10, 8))
# plt.plot(K_range, silhouette_scores, 'o-')
# plt.grid(True)
# plt.show()


optimal_k = 2
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
y_kmeans = kmeans.fit_predict(X_blobs)



# print(X_blobs)
# print(y_true)

plt.figure(figsize=(15,10))
plt.scatter(X_blobs[:, 0], X_blobs[:, 1], c=y_true, alpha=0.6)
plt.show()


plt.figure(figsize=(15,10))
plt.scatter(X_blobs[:, 0], X_blobs[:, 1], c=y_kmeans, alpha=0.6)
plt.show()
