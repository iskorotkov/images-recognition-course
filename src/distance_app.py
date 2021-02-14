import random
from typing import Tuple

import math
from prettytable import PrettyTable

import dataset
import distance

random.seed(0)


def find_closest(item, data, method) -> Tuple[int, float]:
    """
    Find closest object.
    :param item: object for which to find closest object.
    :param data: list of objects to search in.
    :param method: method to use for calculating distances.
    :return: closest point index and distance to the current object.
    """

    closest_index = -1
    min_distance = math.inf
    for i, value in enumerate(data):
        distance = method(item, data[i])
        if distance < min_distance:
            closest_index = i
            min_distance = distance

    return closest_index, min_distance


# Prepare dataset
head, x, labels = dataset.from_file('../data/iris.csv')
x, labels = dataset.shuffle(x, labels)
train_x, test_x = dataset.split(x, ratio=0.9)
train_labels, test_labels = dataset.split(labels, ratio=0.9)

train_len = len(train_x)
test_len = len(test_x)

# Method to use
methods = [
    ('euclidean', distance.euclidean),
    ('hamming', distance.hamming),
    ('manhattan', distance.manhattan),
    ('jaccard', distance.jaccard),
    ('cosine', distance.cosine)
]

# Print header
print('Predict labels for test data\n\n'
      'Find closest object for each item from test data and assign closest object\'s label to it')

print(f'Total train values: {train_len}, test values: {test_len}\n')

# Setup comparison table
comparison_table = PrettyTable(["name", "correct", "total", "percent"])
comparison_table.title = 'Method comparison'

# Iterate over methods
for method_name, method_func in methods:
    # Setup method table
    table = PrettyTable(["n", "predicted", "actual", "correct?", "distance"])
    table.title = f'{method_name} method'

    correct = 0

    # Iterate over samples
    for index, (item, actual_label) in enumerate(zip(test_x, test_labels)):
        closest_index, min_distance = find_closest(item, train_x, method_func)

        closest_item, predicted_label = train_x[closest_index], train_labels[closest_index]
        is_correct = predicted_label == actual_label

        table.add_row([index + 1, predicted_label, actual_label, is_correct, min_distance])
        correct += int(is_correct)

    # Print method results
    print(table)
    print(f'Correct: {correct} out of {test_len} ({correct * 100 / test_len}%)\n')

    comparison_table.add_row([method_name, correct, test_len, f"{correct * 100 / test_len}%"])

print(comparison_table)
