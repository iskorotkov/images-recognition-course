import abc
import random
from typing import List
import numpy
import math
import numpy as np
import distance


class Strategy:
    """
    Base class for all clustering strategies.
    """

    def __init__(self, n_clusters: int = 2) -> None:
        """
        Strategy constructor.
        :param n_clusters: number of clusters in provided datasets.
        """
        super().__init__()
        if n_clusters < 1:
            raise ValueError("number of clusters must be >= 1")

        self.n_clusters = n_clusters
        self.cluster_centers_ = []

    @abc.abstractmethod
    def fit(self, data: List) -> None:
        """
        Train strategy on a provided dataset.
        :param data: dataset to train on.
        """
        raise NotImplementedError

    def predict(self, data: List) -> List[int]:
        """
        Assign dataset items to closest clusters.
        :param data: items to assign to clusters.
        :return: labels (cluster indexes) for each dataset item.
        """
        labels = []
        for item in data:
            nearest_center_index = 0
            min_distance = distance.euclidean(
                item, self.cluster_centers_[nearest_center_index])

            for i, center in enumerate(self.cluster_centers_):
                d = distance.euclidean(item, center)
                if d < min_distance:
                    min_distance = d
                    nearest_center_index = i

            labels.append(nearest_center_index)

        return np.array(labels)


class Maximin(Strategy):
    """
    Maximin clustering strategy.
    """

    def __init__(self, n_clusters: int = 2) -> None:
        super().__init__(n_clusters)
        self.distances_sum_ = 0

    def fit(self, data: List) -> None:
        if len(data) == 0:
            return

        if self.n_clusters == 1:
            return

        if self.n_clusters == len(data):
            return

        if self.n_clusters > len(data):
            self.n_clusters = len(data)

        self.cluster_centers_ = [data[0]]
        while True:
            furthest_item = data[0]
            max_distance = 0

            for item in data:
                distance_to_center = math.inf

                for center in self.cluster_centers_:
                    cur_distance = distance.euclidean(item, center)
                    distance_to_center = min(distance_to_center, cur_distance)

                if distance_to_center > max_distance:
                    max_distance = distance_to_center
                    furthest_item = item

            if self.should_stop(max_distance):
                break
            else:
                self.distances_sum_ += max_distance
                self.cluster_centers_.append(furthest_item)

        return

    def should_stop(self, cur_distance: float) -> bool:
        """
        Determine whether should stop finding clusters.
        :param cur_distance: last calculated maximin distance.
        :return: whether should stop clustering.
        """
        return cur_distance < self.mean_distance() or len(self.cluster_centers_) >= self.n_clusters

    def mean_distance(self) -> float:
        """
        Mean of distances between found cluster centers.
        :return: distances mean.
        """
        return self.distances_sum_ / len(self.cluster_centers_)


class KMaximin(Strategy):
    """
    K-Maximin clustering strategy.
    """

    def __init__(self, n_clusters: int = 2) -> None:
        super().__init__(n_clusters)

    def fit(self, data: List) -> None:
        if len(data) == 0:
            return

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
    K-Means clustering strategy.
    """

    def __init__(self, n_clusters: int = 2) -> None:
        super().__init__(n_clusters)

    def fit(self, data: List) -> None:
        if len(data) == 0:
            return

        if self.n_clusters == 1:
            return

        if self.n_clusters == len(data):
            return

        if self.n_clusters > len(data):
            self.n_clusters = len(data)

        if len(self.cluster_centers_) == 0:
            self.cluster_centers_.append(random.choice(data))
            for _ in range(self.n_clusters - 1):
                weights = self.calc_weights(data)
                self.cluster_centers_.extend(
                    random.choices(data, weights, k=1))

        labels = self.predict(data)
        for i in range(self.n_clusters):
            assigned = [x for x, label in zip(data, labels) if label == i]
            if len(assigned) == 0:
                continue

            self.cluster_centers_[i] = KMeans.mean(assigned)

    def calc_weights(self, data: List) -> List[float]:
        """
        Calculate weights of potential points for k-means++.
        :param data: potential points.
        :return: weights for points.
        """
        weights = []
        for item in data:
            distances_to_centers = map(lambda center: distance.euclidean(
                center, item), self.cluster_centers_)
            min_distance = min(distances_to_centers)
            weights.append(math.pow(min_distance, 2))

        return weights

    @staticmethod
    def mean(assigned: List) -> List:
        """
        Find mean point of the cluster.
        :param assigned: items assigned to the cluster.
        :return: cluster mean point.
        """
        if len(assigned) == 0:
            raise ValueError("list must contain at least 1 element")

        fields = len(assigned[0])
        fields_values = [[item[field_index] for item in assigned]
                         for field_index in range(fields)]
        field_sums = [sum(field_values) for field_values in fields_values]
        center = [field_sum / len(assigned) for field_sum in field_sums]
        return center
