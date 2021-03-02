from typing import List, Tuple

import numpy as np
import math
import dataset


class ConfusionMatrix:
    def __init__(self, tp, tn, fp, fn) -> None:
        self.tp = tp
        self.tn = tn
        self.fp = fp
        self.fn = fn


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


def rocCurve(min_threshold: int, max_threshold: int, rounds: int, evaluate) -> Tuple[List[float], List[float]]:
    """
    Receiver operating characteristic.
    :param min_threshold: Initial threshold value.
    :param max_threshold: Final threshold value.
    :param rounds: Number of iterations.
    :param evaluate: Fit algorithm and predict labels with given threshold; must return tuple of predicted and actual values.
    """
    delta = max_threshold - min_threshold

    x = [0]
    y = [0]
    for i in range(0, rounds+1):
        predicted, actual = evaluate(min_threshold + delta * i / rounds)
        matrix = confusionMatrix(actual, predicted)

        tpr = matrix.tp / (matrix.tp + matrix.fn)
        fpr = matrix.fp / (matrix.fp + matrix.tn)

        # x.append(x[-1] + fpr)
        # y.append(x[-1] + tpr)

        x.append(fpr)
        y.append(tpr)

    return np.array(x) / x[-1], np.array(y) / y[-1]


def mse(actual: List, predicted: List) -> float:
    """
    Root mean square error.
    """
    powers = [math.pow(x-y, 2) for x, y in zip(actual, predicted)]
    s = sum(powers)
    return math.sqrt(s)


def crossValidation(x: List, labels: List, rounds: int, predict, measure) -> float:
    """
    K-fold cross-validation.
    :param x: Dataset.
    :param labels: Data labels.
    :param rounds: Rounds of cross-validation.
    :params predict: Prediction algorithm.
    :params measure: Measuring algorithm.
    """
    x, labels = dataset.shuffle(x, labels)

    rounds = min(rounds, len(x))
    length = len(x)/rounds

    result = 0

    for i in range(rounds):
        start = i * length
        end = start + length

        x_test = x[start:end]
        labels_test = labels[start:end]

        x_train = x[0:start] + x[end:]
        labels_train = labels[0:start] + labels[end:]

        predicted = predict((x_train, labels_train), x_test)
        result += measure(labels_test, predicted)

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


def r_squared():
    # R-squared when no known split available
    pass


def sillhoueteCoef():
    pass


def dunnIndex():
    pass


def dbi():
    # Devis-Boldin index?
    pass


def dbcv():
    # distribution ??? clustering validity?
    pass
