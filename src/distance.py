import math
from typing import List


def euclidean(a: List, b: List) -> float:
    if len(a) != len(b):
        raise ValueError("len(a) != len(b)")

    return math.sqrt(sum(math.pow(x - y, 2) for x, y in zip(a, b)))


def hamming(a: List, b: List) -> float:
    if len(a) != len(b):
        raise ValueError("len(a) != len(b)")

    return sum(x != y for x, y in zip(a, b)) / len(a)


def manhattan(a: List, b: List) -> float:
    if len(a) != len(b):
        raise ValueError("len(a) != len(b)")

    return sum(math.fabs(x - y) for x, y in zip(a, b))


def jaccard(a: List, b: List) -> float:
    if len(a) != len(b):
        raise ValueError("len(a) != len(b)")

    if len(a) == 0:
        return 0

    union = sum([x != 0 or y != 0 for x, y in zip(a, b)])
    intersect = sum([(x != 0 or y != 0) and x == y for x, y in zip(a, b)])
    return 1 - intersect / union


def cosine(a: List, b: List) -> float:
    if len(a) != len(b):
        raise ValueError("len(a) != len(b)")

    ab = sum(x * y for x, y in zip(a, b))
    sa2 = math.sqrt(sum(math.pow(x, 2) for x in a))
    sb2 = math.sqrt(sum(math.pow(y, 2) for y in b))

    similarity = ab / (sa2 * sb2)
    return 1 - similarity
