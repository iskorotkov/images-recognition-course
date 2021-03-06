import math
from typing import List, Tuple
import numpy as np
import distance
import dataset
from scipy.spatial.distance import euclidean, cdist
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csgraph


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
        silhouette = (b - a) / max(a, b)

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
            if min_distance < 0 or dist < min_distance:
                min_distance = dist

    return min_distance / max_diameter


def dbi(data: List, clusters: List, centers: List) -> float:
    """
    Davies-Bouldin index.
    :param data: List of points.
    :param clusters: List of predicted clusters.
    :param centers: List of cluster centers.
    """
    def s(elements: List, center) -> float:
        s = sum([distance.euclidean(x, center) for x in elements])
        length = math.sqrt(sum([x ** 2 for x in center]))
        return s / length

    result = 0
    for ck in range(len(centers)):
        max_value = -1
        for cl in range(len(centers)):
            if ck == cl:
                continue

            ck_center = centers[ck]
            cl_center = centers[cl]
            dist = distance.euclidean(ck_center, cl_center)

            ck_elements = data[[ck == cluster for cluster in clusters]]
            cl_elements = data[[cl == cluster for cluster in clusters]]

            value = (s(ck_elements, ck_center) +
                     s(cl_elements, cl_center)) / dist
            max_value = max(max_value, value)

        result += max_value

    return result / len(data)


def dbcv(X, labels, dist_function=euclidean):
    """
    Density Based clustering validation
    Args:
        X (np.ndarray): ndarray with dimensions [n_samples, n_features]
            data to check validity of clustering
        labels (np.array): clustering assignments for data X
        dist_dunction (func): function to determine distance between objects
            func args must be [np.array, np.array] where each array is a point
    Returns: cluster_validity (float)
        score in range[-1, 1] indicating validity of clustering assignments
    """
    graph = _mutual_reach_dist_graph(X, labels, dist_function)
    mst = _mutual_reach_dist_MST(graph)
    cluster_validity = _clustering_validity_index(mst, labels)
    return cluster_validity


def _core_dist(point, neighbors, dist_function):
    """
    Computes the core distance of a point.
    Core distance is the inverse density of an object.
    Args:
        point (np.array): array of dimensions (n_features,)
            point to compute core distance of
        neighbors (np.ndarray): array of dimensions (n_neighbors, n_features):
            array of all other points in object class
        dist_dunction (func): function to determine distance between objects
            func args must be [np.array, np.array] where each array is a point
    Returns: core_dist (float)
        inverse density of point
    """
    n_features = np.shape(point)[0]
    n_neighbors = np.shape(neighbors)[0]

    distance_vector = cdist(point.reshape(1, -1), neighbors)
    distance_vector = distance_vector[distance_vector != 0]
    numerator = ((1/distance_vector)**n_features).sum()
    core_dist = (numerator / (n_neighbors - 1)) ** (-1/n_features)
    return core_dist


def _mutual_reachability_dist(point_i, point_j, neighbors_i,
                              neighbors_j, dist_function):
    """.
    Computes the mutual reachability distance between points
    Args:
        point_i (np.array): array of dimensions (n_features,)
            point i to compare to point j
        point_j (np.array): array of dimensions (n_features,)
            point i to compare to point i
        neighbors_i (np.ndarray): array of dims (n_neighbors, n_features):
            array of all other points in object class of point i
        neighbors_j (np.ndarray): array of dims (n_neighbors, n_features):
            array of all other points in object class of point j
        dist_dunction (func): function to determine distance between objects
            func args must be [np.array, np.array] where each array is a point
    Returns: mutual_reachability (float)
        mutual reachability between points i and j
    """
    core_dist_i = _core_dist(point_i, neighbors_i, dist_function)
    core_dist_j = _core_dist(point_j, neighbors_j, dist_function)
    dist = dist_function(point_i, point_j)
    mutual_reachability = np.max([core_dist_i, core_dist_j, dist])
    return mutual_reachability


def _mutual_reach_dist_graph(X, labels, dist_function):
    """
    Computes the mutual reach distance complete graph.
    Graph of all pair-wise mutual reachability distances between points
    Args:
        X (np.ndarray): ndarray with dimensions [n_samples, n_features]
            data to check validity of clustering
        labels (np.array): clustering assignments for data X
        dist_dunction (func): function to determine distance between objects
            func args must be [np.array, np.array] where each array is a point
    Returns: graph (np.ndarray)
        array of dimensions (n_samples, n_samples)
        Graph of all pair-wise mutual reachability distances between points.
    """
    n_samples = np.shape(X)[0]
    graph = []
    counter = 0
    for row in range(n_samples):
        graph_row = []
        for col in range(n_samples):
            point_i = X[row]
            point_j = X[col]
            class_i = labels[row]
            class_j = labels[col]
            members_i = _get_label_members(X, labels, class_i)
            members_j = _get_label_members(X, labels, class_j)
            dist = _mutual_reachability_dist(point_i, point_j,
                                             members_i, members_j,
                                             dist_function)
            graph_row.append(dist)
        counter += 1
        graph.append(graph_row)
    graph = np.array(graph)
    return graph


def _mutual_reach_dist_MST(dist_tree):
    """
    Computes minimum spanning tree of the mutual reach distance complete graph
    Args:
        dist_tree (np.ndarray): array of dimensions (n_samples, n_samples)
            Graph of all pair-wise mutual reachability distances
            between points.
    Returns: minimum_spanning_tree (np.ndarray)
        array of dimensions (n_samples, n_samples)
        minimum spanning tree of all pair-wise mutual reachability
            distances between points.
    """
    mst = minimum_spanning_tree(dist_tree).toarray()
    return mst + np.transpose(mst)


def _cluster_density_sparseness(MST, labels, cluster):
    """
    Computes the cluster density sparseness, the minimum density
        within a cluster
    Args:
        MST (np.ndarray): minimum spanning tree of all pair-wise
            mutual reachability distances between points.
        labels (np.array): clustering assignments for data X
        cluster (int): cluster of interest
    Returns: cluster_density_sparseness (float)
        value corresponding to the minimum density within a cluster
    """
    indices = np.where(labels == cluster)[0]
    cluster_MST = MST[indices][:, indices]
    cluster_density_sparseness = np.max(cluster_MST)
    return cluster_density_sparseness


def _cluster_density_separation(MST, labels, cluster_i, cluster_j):
    """
    Computes the density separation between two clusters, the maximum
        density between clusters.
    Args:
        MST (np.ndarray): minimum spanning tree of all pair-wise
            mutual reachability distances between points.
        labels (np.array): clustering assignments for data X
        cluster_i (int): cluster i of interest
        cluster_j (int): cluster j of interest
    Returns: density_separation (float):
        value corresponding to the maximum density between clusters
    """
    indices_i = np.where(labels == cluster_i)[0]
    indices_j = np.where(labels == cluster_j)[0]
    shortest_paths = csgraph.dijkstra(MST, indices=indices_i)
    relevant_paths = shortest_paths[:, indices_j]
    density_separation = np.min(relevant_paths)
    return density_separation


def _cluster_validity_index(MST, labels, cluster):
    """
    Computes the validity of a cluster (validity of assignmnets)
    Args:
        MST (np.ndarray): minimum spanning tree of all pair-wise
            mutual reachability distances between points.
        labels (np.array): clustering assignments for data X
        cluster (int): cluster of interest
    Returns: cluster_validity (float)
        value corresponding to the validity of cluster assignments
    """
    min_density_separation = np.inf
    for cluster_j in np.unique(labels):
        if cluster_j != cluster:
            cluster_density_separation = _cluster_density_separation(MST,
                                                                     labels,
                                                                     cluster,
                                                                     cluster_j)
            if cluster_density_separation < min_density_separation:
                min_density_separation = cluster_density_separation
    cluster_density_sparseness = _cluster_density_sparseness(MST,
                                                             labels,
                                                             cluster)
    numerator = min_density_separation - cluster_density_sparseness
    denominator = np.max([min_density_separation, cluster_density_sparseness])
    cluster_validity = numerator / denominator
    return cluster_validity


def _clustering_validity_index(MST, labels):
    """
    Computes the validity of all clustering assignments for a
    clustering algorithm
    Args:
        MST (np.ndarray): minimum spanning tree of all pair-wise
            mutual reachability distances between points.
        labels (np.array): clustering assignments for data X
    Returns: validity_index (float):
        score in range[-1, 1] indicating validity of clustering assignments
    """
    n_samples = len(labels)
    validity_index = 0
    for label in np.unique(labels):
        fraction = np.sum(labels == label) / float(n_samples)
        cluster_validity = _cluster_validity_index(MST, labels, label)
        validity_index += fraction * cluster_validity
    return validity_index


def _get_label_members(X, labels, cluster):
    """
    Helper function to get samples of a specified cluster.
    Args:
        X (np.ndarray): ndarray with dimensions [n_samples, n_features]
            data to check validity of clustering
        labels (np.array): clustering assignments for data X
        cluster (int): cluster of interest
    Returns: members (np.ndarray)
        array of dimensions (n_samples, n_features) of samples of the
        specified cluster.
    """
    indices = np.where(labels == cluster)[0]
    members = X[indices]
    return members
