import abc
from typing import List

import numpy as np

from src import distance


class Strategy:
    """
    Base class for all classification strategies.
    """

    @abc.abstractmethod
    def fit(self, data: List, labels: List) -> None:
        """
        Train strategy on a provided dataset.
        :param data: Dataset to train on.
        :param labels: Labels for dataset items.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, data: List) -> List:
        """
        Classify items.
        :param data: Items to classify.
        :return: class Labels for each item.
        """
        raise NotImplementedError


class KNN(Strategy):
    """
    k-nearest neighbors classification strategy.
    See kNN (https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm).
    """

    def __init__(self, k: int):
        """
        Creates kNN classifier.
        :param k: Number of nearest neighbors to use.
        """
        self.train = []
        self.k = k

    def fit(self, data: List, labels: List) -> None:
        self.train = list(zip(data, labels))
        self.k = min(self.k, len(self.train))

    def predict(self, data: List) -> List:
        classes = []
        for item in data:
            nearest = sorted(self.train, key=lambda x: distance.euclidean(item, x[0]))
            k_nearest = nearest[:self.k]

            k_classes = list(map(lambda x: x[1], k_nearest))
            nearest_class = max(k_classes, key=lambda x: k_classes.count(x))

            classes.append(nearest_class)

        return np.array(classes)


class SVM(Strategy):
    """
    Support-vector machine classification strategy.

    See Support-vector machine (https://en.wikipedia.org/wiki/Support-vector_machine).
    See Sequential minimal optimization (https://en.wikipedia.org/wiki/Sequential_minimal_optimization).

    Implementation is based on "Yet more simple SMO algorithm"
    (https://www.researchgate.net/publication/344460740_Yet_more_simple_SMO_algorithm).
    Github: https://github.com/fbeilstein/simplest_smo_ever.
    Live demo: https://fbeilstein.github.io/simplest_smo_ever/.
    """

    def __init__(self, target_class, kernel='linear', constraint=10000.0, n_iter=10, degree=3, gamma=1):
        """
        Creates SVM classifier.
        :param target_class: Class to mark as 1 (all other classes will be marked as -1).
        :param kernel: Kernel to use (linear by default - work like in soft margin without a kernel).
        :param constraint: Trade-off between increasing the margin size and ensuring that
        the x_i lie on the correct side of the margin.
        :param n_iter: Number of iteration while fitting.
        :param degree: Degree for polynomial kernel.
        :param gamma: Gamma parameter for RBF (Radial Basis Function) kernel.
        """
        self.kernel = {'poly': lambda x, y: np.dot(x, y.T) ** degree,
                       'rbf': lambda x, y: np.exp(-gamma * np.sum((y - x[:, np.newaxis]) ** 2, axis=-1)),
                       'linear': lambda x, y: np.dot(x, y.T)}[kernel]
        self.target_class = target_class
        self.constraint = constraint
        self.n_iter = n_iter
        self.x = []
        self.y = 0
        self.lambdas = []
        self.k = 0
        self.b = 0

    def restrict_to_square(self, t, v0, u):
        """
        Apply all restrictions to square [0, C] * [0, C].
        :param t: Param t_max from original formula.
        :param v0: Param v0 from original formula.
        :param u: Param u from original formula.
        :return: Return clamped value.
        """
        t = (np.clip(v0 + t * u, 0, self.constraint) - v0)[1] / u[1]
        return (np.clip(v0 + t * u, 0, self.constraint) - v0)[0] / u[0]

    def fit(self, x, y):
        self.x = x
        self.y = np.array(list(map(lambda item: 1 if item == self.target_class else -1, y)), dtype=np.float)
        self.lambdas = np.zeros_like(self.y, dtype=float)
        self.k = self.kernel(self.x, self.x) * self.y[:, np.newaxis] * self.y

        for _ in range(self.n_iter):
            # Freeze all lambdas except lambda_m and lambda_n.

            # Enumerate lambdas (m).
            for iteration in range(len(self.lambdas)):
                # Random index for selecting alpha (l).
                index = np.random.randint(0, len(self.lambdas))

                # Q matrix.
                q = self.k[[[iteration, iteration], [index, index]], [[iteration, index], [iteration, index]]]

                # lambda_m * lambda_l, transposed.
                v0 = self.lambdas[[iteration, index]]

                k0 = 1 - np.sum(self.lambdas * self.k[[iteration, index]], axis=1)
                u = np.array([-self.y[index], self.y[iteration]])
                t_max = np.dot(k0, u) / (np.dot(np.dot(q, u), u) + 1E-15)
                self.lambdas[[iteration, index]] = v0 + u * self.restrict_to_square(t_max, v0, u)

        sv_indices, = np.nonzero(self.lambdas > 1E-15)
        self.b = np.sum((1.0 - np.sum(self.k[sv_indices] * self.lambdas, axis=1)) * self.y[sv_indices]) / len(
            sv_indices)

    def predict(self, data: List) -> List:
        values = np.sum(self.kernel(data, self.x) * self.y * self.lambdas, axis=1) + self.b
        return (values > 0) * 2 - 1
