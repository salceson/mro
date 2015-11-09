# coding=utf-8

from numpy.linalg import LinAlgError
from sklearn import datasets
from sklearn.neighbors import DistanceMetric
import numpy as np
from matplotlib import pyplot as plt

from visualisation_util import plot_2d_classes

__author__ = 'Michał Ciołczyk'

EPSILON = 1e-14


def read_data():
    iris = datasets.load_iris()
    return iris.data[:, :2]


def euclid(_):
    metric = DistanceMetric.get_metric('euclidean')

    def distance(x, y):
        return metric.pairwise([x], [y])[0][0]

    return distance


def mahalonobis(X):
    cov = np.cov(X, rowvar=0)
    try:
        metric = DistanceMetric.get_metric('mahalanobis', V=cov) if X.shape[0] > 1 \
            else DistanceMetric.get_metric('euclidean')
    except LinAlgError:
        metric = DistanceMetric.get_metric('euclidean')

    def distance(x, y):
        return metric.pairwise([x], [y])[0][0]

    return distance


def kMeans(X, k, metric=euclid('')):
    points = np.random.randint(0, X.shape[0], k)
    means = [X[points[i], :] for i in range(k)]
    print(means)
    best_sum = float("inf")
    y = []
    while True:
        y = []
        new_means = [[0 for j in range(2)] for i in range(k)]
        points_num = [0 for i in range(k)]
        for i in range(X.shape[0]):
            min_dist = metric(means[0], X[i, :])
            min_j = 0
            for j in range(1, k):
                dist = metric(means[j], X[i, :])
                if dist < min_dist:
                    min_dist = dist
                    min_j = j
            y.append(min_j)
        print(y)
        for i in range(X.shape[0]):
            for j in range(2):
                new_means[y[i]][j] += X[i][j]
            points_num[y[i]] += 1
        print(points_num)
        for i in range(k):
            for j in range(2):
                if points_num[i] > 0:
                    new_means[i][j] /= float(points_num[i])
        sums = [0 for i in range(k)]
        for i in range(X.shape[0]):
            sums[y[i]] += metric(new_means[y[i]], X[i, :])
        sum = np.sum(sums)
        if sum - best_sum > -EPSILON:
            break
        best_sum = sum
        means = new_means
        print(means)

    return means, y


if __name__ == '__main__':
    X = read_data()
    k = 3
    means, y = kMeans(X, k)
    plot_2d_classes(X, np.array(y), 'ryb')
    colors = 'ryb'
    for i in range(k):
        [x, y] = means[i]
        plt.plot(x, y, '^', c=colors[i])
    plt.show()
