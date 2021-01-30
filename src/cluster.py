import abc
import random
from typing import List

import distance


class Strategy:
    def __init__(self, n_clusters: int = 2) -> None:
        super().__init__()
        if n_clusters < 1:
            raise ValueError("number of clusters must be >= 1")

        self.n_clusters = n_clusters
        self.cluster_centers = []

    @abc.abstractmethod
    def fit(self, data: List) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, data: List) -> List[int]:
        raise NotImplementedError


class Maximin(Strategy):
    def __init__(self, n_clusters: int = 2) -> None:
        super().__init__(n_clusters)

    def fit(self, data: List) -> None:
        if self.n_clusters == 1:
            return

        if self.n_clusters == len(data):
            return

        if self.n_clusters > len(data):
            self.n_clusters = len(data)

        self.cluster_centers = [data[0]]
        for _ in range(self.n_clusters - 1):
            furthest_item = max(data, key=lambda item: min(distance.euclidean(item, center)
                                                           for center in self.cluster_centers))
            self.cluster_centers.append(furthest_item)

        return

    def predict(self, data: List) -> List[int]:
        labels = []
        for item in data:
            nearest_cluster = min(self.cluster_centers, key=lambda center: distance.euclidean(item, center))
            labels.append(self.cluster_centers.index(nearest_cluster))

        return labels


class KMeans(Strategy):
    def __init__(self, n_clusters: int = 2) -> None:
        super().__init__(n_clusters)

    def fit(self, data: List) -> None:
        if self.n_clusters == 1:
            return

        if self.n_clusters == len(data):
            return

        if self.n_clusters > len(data):
            self.n_clusters = len(data)

        if len(self.cluster_centers) == 0:
            self.cluster_centers = random.choices(data, k=self.n_clusters)

        labels = self.predict(data)
        for i in range(self.n_clusters):
            assigned = [x for x, label in zip(data, labels) if label == i]
            if len(assigned) == 0:
                continue

            self.cluster_centers[i] = KMeans.mean(assigned)

    def predict(self, data: List) -> List[int]:
        labels = []
        for item in data:
            nearest_cluster = min(self.cluster_centers, key=lambda center: distance.euclidean(item, center))
            labels.append(self.cluster_centers.index(nearest_cluster))

        return labels

    @staticmethod
    def mean(assigned: List):
        if len(assigned) == 0:
            raise ValueError("list must contain at least 1 element")

        fields = len(assigned[0])
        fields_values = [[item[field_index] for item in assigned] for field_index in range(fields)]
        field_sums = [sum(field_values) for field_values in fields_values]
        center = [field_sum / len(assigned) for field_sum in field_sums]
        return center
