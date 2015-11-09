# coding=utf-8
from queue import Queue

import random
import operator
import sys
from numpy.linalg import LinAlgError

from sklearn import datasets
from sklearn.neighbors import DistanceMetric
import numpy as np
import matplotlib.pyplot as plt

from visualisation_util import plot_areas, plot_2d_classes

__author__ = 'Michał Ciołczyk'

SHOW_PREDICTIONS_AND_REAL_VALUES = False


def read_data(only_2_features=True):
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    if only_2_features:
        X = X[:, :2]
    return X, y


def split_dataset(X, y, ratio):
    X_training = []
    y_training = []
    X_test = []
    y_test = []
    for i in range(X.shape[0]):
        if random.random() < ratio:
            X_training.append(X[i, :])
            y_training.append(y[i])
        else:
            X_test.append(X[i, :])
            y_test.append(y[i])
    return np.array(X_training), np.array(y_training), np.array(X_test), np.array(y_test)


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


def kNN(X, X_training, y_training, test_instance, k, metric=euclid):
    metric = metric(X)
    distances = []
    for i in range(X_training.shape[0]):
        distances.append((y_training[i], metric(X_training[i, :], test_instance)))
    distances.sort(key=operator.itemgetter(1))
    neighbours = list(map(operator.itemgetter(0), distances[:k]))
    votes = {}
    for i in range(len(neighbours)):
        response = neighbours[i]
        if response in votes:
            votes[response] += 1
        else:
            votes[response] = 1
    return sorted(votes, reverse=True)[0]


def cnn_transform(X, y, k, metric):
    X_store = []
    y_store = []
    bag = Queue()
    for i in range(X.shape[0]):
        bag.put((X[i, :], y[i]))
    p = bag.get()
    X_store.append(p[0])
    y_store.append(p[1])
    n = 0
    while not bag.empty() and n < bag.qsize():
        n += 1
        p = bag.get()
        X_step = np.array(X_store)
        if kNN(X_step, X_step, np.array(y_store), np.array(p[0]), k, metric) == p[1]:
            bag.put(p)
        else:
            X_store.append(p[0])
            y_store.append(p[1])
            n = 0
    return np.array(X_store), np.array(y_store)


if __name__ == '__main__':
    k = int(sys.argv[1])
    cnn = sys.argv[2]
    cnn = True if cnn == '1' else False
    metric = sys.argv[3]
    mode = sys.argv[4]
    mode = False if mode == 'A' else True
    X, y = read_data(not mode)
    metric = euclid if metric == 'euclid' else mahalonobis
    if mode:
        accuracies = []
        cnn_percents = []
        for i in range(10):
            X_training, y_training, X_test, y_test = split_dataset(X, y, 0.7)
            if cnn:
                len_before = X_training.shape[0]
                X_training, y_training = cnn_transform(X_training, y_training, k, metric)
                len_after = X_training.shape[0]
                cnn_percents.append(float(len_after) / float(len_before) * 100.0)
            predictions = []
            for i in range(X_test.shape[0]):
                predictions.append(kNN(X_training, X_training, y_training, X_test[i, :], k, metric))
            if SHOW_PREDICTIONS_AND_REAL_VALUES:
                print('Prediction, actual:')
                for i in range(X_test.shape[0]):
                    print(predictions[i], y_test[i])
            correct = 0
            for i in range(len(predictions)):
                if y_test[i] == predictions[i]:
                    correct += 1
            accuracies.append(float(correct) / float(len(predictions)) * 100.0)
        print("Accuracy:", str(np.mean(accuracies)) + '%', 'StdDev:', np.std(accuracies))
        if cnn:
            print("CNN:", str(np.mean(cnn_percents)) + '%', 'StdDev:', np.std(cnn_percents))
    else:
        if cnn:
            X, y = cnn_transform(X, y, k, metric)
        plot_areas(lambda x: kNN(X, X, y, x, k, metric), 0.1, X)
        plot_2d_classes(X, y, 'ryb')
        cnn_str = '_cnn' if cnn else ''
        plt.savefig('plots/k' + str(k) + '_' + metric.__name__ + cnn_str + '.png')
