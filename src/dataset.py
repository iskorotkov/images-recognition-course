import random
from typing import List, Tuple

import pandas as pd


def from_file(path: str) -> Tuple[List, List, List]:
    """
    Load dataset from file.
    :param path: path to dataset.
    :return: header values, data, associated labels.
    """
    data = pd.read_csv(path)

    head = data.head(n=1)
    x = data.iloc[:, :-1].values
    labels = data.iloc[:, -1]

    return head, x, labels


def split(x: List, ratio: float = 0.9) -> Tuple[List, List]:
    """
    Split provided dataset into two parts.
    :param x: dataset to split.
    :param ratio: split ratio.
    :return: (first part, second part).
    """
    edge = int(len(x) * ratio)
    return x[:edge], x[edge:]


def shuffle(x: List, labels: List) -> Tuple[List, List]:
    """
    Shuffle dataset and labels.
    :param x: dataset values.
    :param labels: dataset labels.
    :return: shuffled dataset and labels.
    """
    if len(x) != len(labels):
        raise ValueError("len(x) != len(labels)")

    indices = list(range(len(x)))
    random.shuffle(indices)
    return [x[i] for i in indices], [labels[i] for i in indices]
