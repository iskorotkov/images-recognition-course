import random
from typing import List, Tuple, Any

import numpy as np
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from sklearn.decomposition import PCA

from src import dataset, classification


def map_color(x: Any):
    """
    Map class label to a color.
    :param x: Class label to map.
    :return: Color value of specified class.
    """
    return {
        -1: 'grey',
        1: 'yellow',
        "setosa": 'yellow',
        "versicolor": 'green',
        "virginica": 'blue'
    }[x]


# Seed
random.seed(0)
np.random.seed(0)

# Prepare dataset
head, x, labels = dataset.from_file('../data/iris.csv')
x, labels = dataset.shuffle(x, labels)
train_x, test_x = dataset.split(x, ratio=0.9)
train_labels, test_labels = dataset.split(labels, ratio=0.9)

# Setup PCA
pca = PCA(2)
pca.fit(x)
transformed = pca.transform(test_x)

# Print header
print('Predict labels for test data\n')
print(f'Total train values: {len(train_x)}, test values: {len(test_x)}\n')

# Methods to use
methods: List[Tuple[str, classification.Strategy]] = [
    ('kNN-1', classification.KNN(1)),
    ('kNN-3', classification.KNN(3)),
    ('kNN-5', classification.KNN(5)),
    ('SVM (setosa)', classification.SVM('setosa')),
    ('SVM (versicolor)', classification.SVM('versicolor')),
    ('SVM (virginica)', classification.SVM('virginica')),
]

# Setup comparison table
comparison_table = PrettyTable()
comparison_table.add_column("Actual", test_labels)

# Setup graph
plt.figure(figsize=(12, 6))
plt.suptitle('Classification methods')
plt.subplot(2, 4, 1)
plt.title('Actual')
plt.scatter(transformed[:, 0], transformed[:, 1], c=list(map(map_color, test_labels)))

# Iterate over methods
for i, (name, method) in enumerate(methods):
    method.fit(train_x, train_labels)
    predicted = method.predict(test_x)

    comparison_table.add_column(name, predicted)

    # Add plot
    plt.subplot(2, 4, i + 2)
    plt.title(name)
    plt.scatter(transformed[:, 0], transformed[:, 1], c=list(map(map_color, predicted)))

print(comparison_table)
plt.show()
