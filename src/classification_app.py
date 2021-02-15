import random
from typing import List, Tuple

from prettytable import PrettyTable

from src import dataset, classification

# Seed
random.seed(0)

# Prepare dataset
head, x, labels = dataset.from_file('../data/iris.csv')
x, labels = dataset.shuffle(x, labels)
train_x, test_x = dataset.split(x, ratio=0.9)
train_labels, test_labels = dataset.split(labels, ratio=0.9)

train_len = len(train_x)
test_len = len(test_x)

# Methods to use
methods: List[Tuple[str, classification.Strategy]] = [
    ('kNN-1', classification.KNN(1)),
    ('kNN-3', classification.KNN(3)),
    ('kNN-5', classification.KNN(5)),
    ('SVM (setosa)', classification.SVM('setosa')),
    ('SVM (versicolor)', classification.SVM('versicolor')),
    ('SVM (virginica)', classification.SVM('virginica')),
]

# Print header
print('Predict labels for test data\n')
print(f'Total train values: {train_len}, test values: {test_len}\n')

# Setup comparison table
comparison_table = PrettyTable()
comparison_table.add_column("Actual", test_labels)

# Iterate over methods
for name, method in methods:
    method.fit(train_x, train_labels)
    predicted = method.predict(test_x)
    comparison_table.add_column(name, predicted)

print(comparison_table)
