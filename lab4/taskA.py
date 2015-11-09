# coding=utf-8

import csv

import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.utils import shuffle

__author__ = 'Michał Ciołczyk'


def read_wines():
    X = []
    y = []
    with open('wine.data', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            y.append(int(row[0]) - 1)
            X.append([float(row[i + 1]) for i in range(13)])
    X = np.array(X)
    y = np.array(y)
    scaler = StandardScaler().fit(X)
    return scaler.transform(X), y


def read_digits():
    d = load_digits()
    X, y = d.data, d.target
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    return X, y


class AdaBoostClassifier(object):
    def __init__(self, classifier, iterations, classifier_params, X_test, y_test):
        self.classifier = classifier
        self.classifier_params = classifier_params
        self.iterations = iterations
        self.classifiers = []
        self.alphas = []
        self.errors = []
        self.fitted = False
        self.X_test = X_test
        self.y_test = y_test

    def fit(self, X_train, y_train):
        n = len(X_train)
        weights = [1.0 / n for _ in range(n)]
        k = len(np.unique(y_train))
        for i in range(self.iterations):
            current_classifier = self.classifier(**self.classifier_params)
            current_classifier = current_classifier.fit(X_train, y_train,
                                                        sample_weight=[weights[i] for i in range(n)])
            self.classifiers.append(current_classifier)

            predicted = [current_classifier.predict(x) for x in X_train]

            errors = np.sum([1 if current_classifier.predict(self.X_test[i, :]) != self.y_test[i] else 0
                             for i in range(len(self.X_test))])
            self.errors.append(errors / float(len(self.y_test)))

            weighted_errors = [weights[i] if y_train[i] != predicted[i] else 0 for i in range(n)]
            error = np.sum(weighted_errors) / np.sum(weights)

            alpha = np.log((1.0 - error) / error) + np.log(k - 1)
            self.alphas.append(alpha)

            weights = [weights[i] * np.exp(-alpha) if y_train[i] == predicted[i]
                       else weights[i] * np.exp(alpha) for i in range(n)]
            sum_weights = np.sum(weights)
            weights = [weights[i] / sum_weights for i in range(n)]
        self.fitted = True
        return self

    def predict(self, x):
        if not self.fitted:
            raise AssertionError('Tried to predict before fitting!')
        predicted = [self.classifiers[i].predict(x)[0] for i in range(len(self.classifiers))]
        classes = {}
        for i in range(len(predicted)):
            if predicted[i] in classes:
                classes[predicted[i]] = classes[predicted[i]] + self.alphas[i]
            else:
                classes[predicted[i]] = self.alphas[i]
        max = float("-inf")
        max_i = -1
        for c in classes:
            if classes[c] > max:
                max = classes[c]
                max_i = c
        return max_i


if __name__ == '__main__':
    datasets = [read_wines, read_digits]
    iterations = 50
    for dataset in datasets:
        dataset_name = dataset.__name__[5:]
        print('Processing dataset:', dataset_name)
        X, y = dataset()
        X, y = shuffle(X, y, random_state=0)
        n = len(X)
        n_train = int(0.7 * n)
        X_train = X[:n_train, :]
        y_train = y[:n_train]
        X_test = X[n_train:, :]
        y_test = y[n_train:]
        svm = SVC(kernel='rbf', C=5)
        svm = svm.fit(X_train, y_train)
        svm_error = np.sum([1 if svm.predict(X_test[i, :]) != y_test[i] else 0
                            for i in range(len(X_test))]) / float(len(X_test))
        ada_boost = AdaBoostClassifier(SVC, iterations, {'kernel': 'rbf', 'C': 5}, X_test, y_test)
        ada_boost.fit(X_train, y_train)
        predicted_svm = np.array([svm.predict(x)[0] for x in X_test])
        predicted_ada = np.array([ada_boost.predict(x) for x in X_test])
        plt.title('Dataset: ' + dataset_name)
        plt.plot(range(iterations), [1 - e for e in ada_boost.errors], c='b',
                 label='Ada Boost (with SAMME)', linewidth=2)
        plt.plot(range(iterations), [1 - svm_error for _ in range(iterations)],
                 c='r', label='SVM (rbf kernel, C=5)', linewidth=2)
        plt.xlabel('iterations')
        plt.ylabel('accuracy')
        plt.legend(loc=0)
        plt.savefig('A_' + dataset_name + '.png')
        plt.figure()
