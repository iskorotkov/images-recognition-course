from typing import List, Tuple

import pandas as pd


def from_file(path: str) -> Tuple[List, List, List]:
    data = pd.read_csv(path)

    head = data.head(n=1)
    x = data.iloc[:, :-1].values
    labels = data.iloc[:, -1]

    return head, x, labels


def split(x: List, labels: List, ratio: float = 0.9) -> Tuple[Tuple[List, List], Tuple[List, List]]:
    if len(x) != len(labels):
        raise ValueError("len(x) != len(labels)")

    edge = int(len(x) * ratio)
    return (x[:edge], labels[:edge]), (x[edge:], labels[edge:])
