import random

import sklearn.cluster as sklearn
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from src import dataset, cluster

head, x, labels = dataset.from_file('../data/iris.csv')

n_clusters = 7
n_iterations = 5


def show_clusters(title: str, method, data, n_clusters: int = 3, n_iterations: int = 1):
    plt.title(title)

    km = method(n_clusters=n_clusters)
    for _ in range(n_iterations):
        km.fit(data)

    labels = km.predict(data)

    pca = PCA(2)

    data = pca.fit_transform(data)
    plt.scatter(data[:, 0], data[:, 1], c=labels)

    centers = pca.transform(km.cluster_centers_)
    plt.scatter(centers[:, 0], centers[:, 1], c=list(range(len(centers))), s=1000, alpha=0.25)


random.seed(0)

plt.figure(figsize=(12, 6))
plt.suptitle('clusters')

plt.subplot(2, 3, 1)
show_clusters('custom maximin', cluster.Maximin, x, n_clusters)

plt.subplot(2, 3, 2)
show_clusters('custom k-means', cluster.KMeans, x, n_clusters)

plt.subplot(2, 3, 3)
show_clusters('sklearn k-means', sklearn.KMeans, x, n_clusters)

plt.subplot(2, 3, 5)
show_clusters(f'custom k-means ({n_iterations} iterations)', cluster.KMeans, x, n_clusters, n_iterations)

plt.subplot(2, 3, 6)
show_clusters(f'sklearn k-means ({n_iterations} iterations)', sklearn.KMeans, x, n_clusters, n_iterations)

plt.show()
