import random
from typing import Tuple, List

import matplotlib.pyplot as plt
import sklearn.cluster as sklearn

from src import cluster

data = [
    [1, 1],
    [1, 2],
    [2, 1],
    [0.5, 1],
    [1, 1.5],
    [1.2, 1],
    [3, 4],
    [4, 3],
    [4, 4],
    [3.4, 4],
    [4, 3.2],
    [3.8, 4],
]


def xy(items: List) -> Tuple[List[float], List[float]]:
    return [x for x, _ in items], [y for _, y in items]


def show_clusters(title: str, method, data, n_clusters: int = 2):
    plt.title(title)

    km = method(n_clusters=n_clusters)
    km.fit(data)
    labels = km.predict(data)

    x, y = xy(data)
    for x, y, label in zip(x, y, labels):
        colors = ['r', 'g', 'b']
        selected_color = colors[label % len(colors)]
        plt.plot(x, y, f'o{selected_color}')


random.seed(0)

plt.figure(figsize=(12, 4))
plt.suptitle('clusters')

plt.subplot(1, 3, 1)
show_clusters('custom maximin', cluster.Maximin, data)

plt.subplot(1, 3, 2)
show_clusters('custom k-means', cluster.KMeans, data)

plt.subplot(1, 3, 3)
show_clusters('sklearn k-means', sklearn.KMeans, data)

plt.show()
