import tensorflow as tf
import metrics
import dataset
import classification
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import tensorflow as tf
import clustering

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

seed = 5
epochs = 2
target = 'virginica'

# Seed
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# Prepare dataset
head, x, labels = dataset.from_file('./data/iris.csv')
x, labels = dataset.shuffle(x, labels)

labels = np.array([1 if x == target else 0 for x in labels])

x_train, x_test = dataset.split(x, ratio=0.9)
y_train, y_test = dataset.split(labels, ratio=0.9)

# Prepare model
model = classification.NN(epochs)
model.fit(x_train, y_train)

# Predict labels
predicted = model.predict(x_test)
predicted_labels = predicted > 0.5

# Evaluate external metrics
precision, recall = metrics.precision_recall(y_test, predicted_labels)
print('Precision =', precision, ', recall =', recall)

f1 = metrics.f1(y_test, predicted_labels)
print('F1 score =', f1)

matrix = metrics.confusionMatrix(y_test, predicted_labels)
print(matrix)

mse = metrics.mse(y_test, predicted)
print('MSE =', mse)


def cross_validation_predict(train, x_test):
    model = classification.NN(epochs)

    x, y = train
    model.fit(x, y)
    return model.predict(x_test) > 0.5


cross_val = metrics.crossValidation(x, labels, 5, cross_validation_predict)
print('Cross validation - correct labels in average =', cross_val)


def roc_predict(threshold: float):
    return model.predict(x_test) + threshold, y_test


fpr, tpr = metrics.rocCurve(100, roc_predict)
plt.title('ROC curve')
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], 'r--')
plt.show()

r2 = metrics.r_squared(y_test, predicted)
print('Coefficient of determination (R squared) =', r2)

kmeans = clustering.KMeans(3)
kmeans.fit(x)
predicted_clusters = kmeans.predict(x)

silhouette = metrics.silhouetteCoef(x, predicted_clusters, kmeans.cluster_centers_)
print('Silhouette (average) =', sum(silhouette) / len(silhouette))

dunnIndex = metrics.dunnIndex(x, predicted_clusters, kmeans.cluster_centers_)
print('Dunn index =', dunnIndex)

dbi = metrics.dbi(x, predicted_clusters, kmeans.cluster_centers_)
print('Devies-Bouldin index =', dbi)

dbcv = metrics.dbcv(x, predicted_clusters)
print('Density-based clustering validation =', dbcv)
