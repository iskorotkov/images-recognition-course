import abc
import random
from typing import List

import distance


class Strategy:
    """
    Base class for all clustering strategies
    """

    def __init__(self, n_clusters: int = 2) -> None:
        """
        Strategy constructor
        :param n_clusters: number of clusters in provided datasets
        """
        super().__init__()
        if n_clusters < 1:
            raise ValueError("number of clusters must be >= 1")

        self.n_clusters = n_clusters
        self.cluster_centers_ = []

    @abc.abstractmethod
    def fit(self, data: List) -> None:
        """
        Train strategy on a provided dataset
        :param data: dataset to train on
        """
        raise NotImplementedError

    def predict(self, data: List) -> List[int]:
        """
        Assign dataset items to closest clusters
        :param data: items to assign to clusters
        :return: labels (cluster indexes) for each dataset item
        """
        labels = []
        for item in data:
            nearest_center_index = 0
            min_distance = distance.euclidean(item, self.cluster_centers_[nearest_center_index])

            for i, center in enumerate(self.cluster_centers_):
                d = distance.euclidean(item, center)
                if d < min_distance:
                    min_distance = d
                    nearest_center_index = i

            labels.append(nearest_center_index)

        return labels


class Maximin(Strategy):
    """
    Maximin clustering strategy
    """

    def __init__(self, n_clusters: int = 2) -> None:
        super().__init__(n_clusters)

    def fit(self, data: List) -> None:
        if self.n_clusters == 1:
            return

        if self.n_clusters == len(data):
            return

        if self.n_clusters > len(data):
            self.n_clusters = len(data)

        self.cluster_centers_ = [data[0]]
        for _ in range(self.n_clusters - 1):
            furthest_item = max(data, key=lambda item: min(distance.euclidean(item, center)
                                                           for center in self.cluster_centers_))
            self.cluster_centers_.append(furthest_item)

        return


class KMeans(Strategy):
    """
    K-Means clustering strategy
    """

    def __init__(self, n_clusters: int = 2) -> None:
        super().__init__(n_clusters)

    def fit(self, data: List) -> None:
        if self.n_clusters == 1:
            return

        if self.n_clusters == len(data):
            return

        if self.n_clusters > len(data):
            self.n_clusters = len(data)

        if len(self.cluster_centers_) == 0:
            self.cluster_centers_ = random.choices(data, k=self.n_clusters)

        labels = self.predict(data)
        for i in range(self.n_clusters):
            assigned = [x for x, label in zip(data, labels) if label == i]
            if len(assigned) == 0:
                continue

            self.cluster_centers_[i] = KMeans.mean(assigned)

    @staticmethod
    def mean(assigned: List) -> List:
        """
        Find mean point of the cluster
        :param assigned: items assigned to the cluster
        :return: cluster mean point
        """
        if len(assigned) == 0:
            raise ValueError("list must contain at least 1 element")

        fields = len(assigned[0])
        fields_values = [[item[field_index] for item in assigned] for field_index in range(fields)]
        field_sums = [sum(field_values) for field_values in fields_values]
        center = [field_sum / len(assigned) for field_sum in field_sums]
        return center
