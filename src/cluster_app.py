import random

import sklearn.cluster as sklearn
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from src import dataset, cluster

# Parameters to tweak
n_clusters = 7
n_iterations = 5

# Prepare dataset
head, x, labels = dataset.from_file('../data/iris.csv')


def show_clusters(title: str, method, data, n_clusters: int = 3, n_iterations: int = 1) -> None:
    """
    Show clusters on a graph
    :param title: graph title
    :param method: method for finding clusters
    :param data: data to search clusters in
    :param n_clusters: number of clusters
    :param n_iterations: number of iterations of the specified method
    """

    plt.title(title)

    # Train method
    km = method(n_clusters=n_clusters)
    for _ in range(n_iterations):
        km.fit(data)

    # Assign to clusters
    labels = km.predict(data)

    # Project multidimensional data onto a plane
    pca = PCA(2)
    data = pca.fit_transform(data)
    centers = pca.transform(km.cluster_centers_)

    # Show points and cluster centers
    plt.scatter(data[:, 0], data[:, 1], c=labels)
    plt.scatter(centers[:, 0], centers[:, 1], c=list(range(len(centers))), s=1000, alpha=0.25)


random.seed(0)

# Setup figure
plt.figure(figsize=(12, 6))
plt.suptitle('clusters')

# Add graphs for each method
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
