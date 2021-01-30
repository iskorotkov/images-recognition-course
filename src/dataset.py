import pandas as pd


def from_file(path):
    data = pd.read_csv(path)

    head = data.head(n=1)
    x = data.iloc[:, :-1].values
    labels = data.iloc[:, -1]

    return head, x, labels


def split(x, labels, ratio=0.9):
    if len(x) != len(labels):
        raise ValueError("len(x) != len(labels)")

    edge = int(len(x) * ratio)
    return (x[:edge], x[edge:]), (labels[:edge], labels[edge:])
