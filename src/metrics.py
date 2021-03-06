import math
from typing import List, Tuple
import numpy as np
import distance
import dataset


class ConfusionMatrix:
    def __init__(self, tp: int, tn: int, fp: int, fn: int) -> None:
        self.tp = tp
        self.tn = tn
        self.fp = fp
        self.fn = fn

    def __repr__(self) -> str:
        return '<ConfusionMatrix tp:%d tn:%d fp:%d fn:%d>' % (self.tp, self.tn, self.fp, self.fn)

    def __str__(self) -> str:
        return 'Confusion matrix: true positives = %d; true negatives = %d; false positives = %d; false negatives = %d' % (self.tp, self.tn, self.fp, self.fn)


def precision_recall(actual: List, predicted: List) -> Tuple[float, float]:
    """
    Returns (precision, recall) values.
    """
    m = confusionMatrix(actual, predicted)
    return m.tp / (m.tp + m.fp), m.tp / (m.tp + m.fn)


def f1(actual: List, predicted: List) -> float:
    """
    F1 measure.
    """
    precision, recall = precision_recall(actual, predicted)
    return 2 * precision * recall / (precision + recall)


def rocCurve(rounds: int, predict, min_threshold: int = -0.5, max_threshold: int = 0.5) -> Tuple[List[float], List[float]]:
    """
    Receiver operating characteristic.
    :param rounds: Number of iterations.
    :param predict: Fit algorithm and predict labels with given threshold; must return tuple of predicted and actual values.
    :param min_threshold: Initial threshold value.
    :param max_threshold: Final threshold value.
    """
    delta = max_threshold - min_threshold

    x = [0]
    y = [0]
    for i in range(0, rounds+1):
        predicted, actual = predict(min_threshold + delta * i / rounds)
        predicted_labels = predicted > 0.5
        matrix = confusionMatrix(actual, predicted_labels)

        tpr = matrix.tp / (matrix.tp + matrix.fn)
        fpr = matrix.fp / (matrix.fp + matrix.tn)

        x.append(fpr)
        y.append(tpr)

    return np.array(x) / x[-1], np.array(y) / y[-1]


def mse(actual: List, predicted: List) -> float:
    """
    Mean square error.
    """
    powers = [math.pow(x-y, 2) for x, y in zip(actual, predicted)]
    return sum(powers) / len(powers)


def crossValidation(x: List, labels: List, rounds: int, predict) -> float:
    """
    K-fold cross-validation.
    :param x: Dataset.
    :param labels: Data labels.
    :param rounds: Rounds of cross-validation.
    :params predict: Prediction algorithm.
    """
    x, labels = dataset.shuffle(x, labels)

    rounds = min(rounds, len(x))
    length = len(x)//rounds

    result = 0

    for i in range(rounds):
        start = i * length
        end = start + length

        x_test = x[start:end]
        labels_test = labels[start:end]

        x_train = np.concatenate((x[:start], x[end:]))
        labels_train = np.concatenate((labels[:start], labels[end:]))

        predicted = predict((x_train, labels_train), x_test)
        result += sum(predicted == labels_test) / len(predicted)

    return result/rounds


def confusionMatrix(actual: List, predicted: List) -> ConfusionMatrix:
    """
    Matrix representing true positives, true negatives, false positives and false negatives.
    """
    m = ConfusionMatrix(0, 0, 0, 0)
    for actual, predicted in zip(actual, predicted):
        if predicted:
            if actual == predicted:
                m.tp += 1
            else:
                m.fp += 1
        else:
            if actual == predicted:
                m.tn += 1
            else:
                m.fn += 1
    return m


def r_squared(actual: List, predicted: List) -> float:
    """
    Coefficient of determination (R squared).
    """
    rss = sum([(y - yp) ** 2 for y, yp in zip(actual, predicted)])
    y_mean = sum(actual) / len(actual)
    tss = sum([(y - y_mean) ** 2 for y in actual])
    return 1 - rss / tss


def silhouetteCoef(data: List, clusters: List, centers: List) -> List[float]:
    """
    Silhouette = `(b - a) / max(a, b)`,
    where `a` - mean distance from the point to other points on the same cluster,
    `b` - mean distance from the point to points from other closest cluster.
    :param data: List of points.
    :param clusters: List of predicted clusters.
    :param centers: List of cluster centers.
    """
    results = []
    for cur_elem, cur_cluster in zip(data, clusters):
        indices = [cluster == cur_cluster for cluster in clusters]
        same_cluster_elems = data[indices]

        center = sorted(
            centers, key=lambda center: distance.euclidean(cur_elem, center))[1]
        center_index = centers.index(center)
        indices = [cluster == center_index for cluster in clusters]
        closest_cluster_elems = data[indices]

        same_cluster_distances = [distance.euclidean(
            cur_elem, other) for other in same_cluster_elems]
        closest_cluster_distances = [distance.euclidean(
            cur_elem, other) for other in closest_cluster_elems]

        # Use (len - 1) because current element is in same_cluster_distances.
        a = sum(same_cluster_distances) / (len(same_cluster_distances) - 1)
        b = sum(closest_cluster_distances) / len(closest_cluster_distances)
        silhouette = abs(b - a) / max(a, b)

        results.append(silhouette)

    return np.array(results)


def dunnIndex(data: List, clusters: List, centers: List) -> float:
    """
    Dunn index.
    :param data: List of points.
    :param clusters: List of predicted clusters.
    :param centers: List of cluster centers.
    """
    def cluster_distance(ck_elems: List, cl_elems: List) -> float:
        distances = [distance.euclidean(x, y)
                     for x in ck_elems
                     for y in cl_elems]
        return min(distances)

    def cluster_diameter(data: List) -> float:
        distances = [distance.euclidean(data[i], data[j])
                     for i in range(len(data))
                     for j in range(i + 1, len(data))]
        return max(distances)

    min_distance = -1
    max_diameter = -1
    for ck in range(len(centers)):
        ck_elems = data[[ck == cluster for cluster in clusters]]

        diameter = cluster_diameter(ck_elems)
        max_diameter = max(max_diameter, diameter)

        for cl in range(ck + 1, len(centers)):
            cl_elems = data[[cl == cluster for cluster in clusters]]

            dist = cluster_distance(ck_elems, cl_elems)
            if (min_distance < 0 or dist < min_distance):
                min_distance = dist

    return min_distance / max_diameter


def dbi():
    # Devis-Boldin index?
    pass


def dbcv():
    # distribution ??? clustering validity?
    pass
