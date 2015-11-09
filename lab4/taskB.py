# coding=utf-8
import csv

import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
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


class StackingClassifier(object):
    def __init__(self, classifiers):
        self.classifiers = classifiers
        self.fitted = False
        self.classifier = LinearRegression()

    def fit(self, X, y):
        for classifier in self.classifiers:
            classifier.fit(X, y)
        X_to_aggregation = np.array([[classifier.predict(x)[0] for classifier in self.classifiers] for x in X])
        self.classifier.fit(X_to_aggregation, y)
        self.fitted = True
        return self

    def predict(self, x):
        if not self.fitted:
            raise AssertionError('Tried to predict before fitting!')
        return [int(round(self.classifier.predict([classifier.predict(x)[0]
                                                   for classifier in self.classifiers])[0], 0))]


def classifier_error(classifier, X_test, y_test):
    return np.sum([1 if classifier.predict(X_test[i, :])[0] != y_test[i] else 0
                   for i in range(len(X_test))]) / float(len(X_test))


def plot_confusion_matrix(cm, label, ax=None, cmap=plt.cm.Blues):
    if not ax:
        ax = plt.gca()
    img = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title('Confusion matrix:\n' + label, fontsize=10)
    plt.colorbar(img, ax=ax)
    tick_marks = range(len(cm))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(tick_marks)
    ax.set_ylabel('True label', fontsize=8)
    ax.set_xlabel('Predicted label', fontsize=8)


if __name__ == '__main__':
    datasets = [read_wines, read_digits]
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
        classifiers = [SVC(), SVC(C=2.0), SVC(kernel='poly', degree=2), SVC(kernel='poly', degree=2, C=2.0),
                       KNeighborsClassifier(n_neighbors=1), KNeighborsClassifier(n_neighbors=1, p=1),
                       KNeighborsClassifier(n_neighbors=5), KNeighborsClassifier(n_neighbors=5, p=1)]
        classifiers_labels = ['SVM (C=1, kernel=rbf)', 'SVM (C=2, kernel=rbf)',
                              'SVM (C=1, kernel=polynomial; 2)', 'SVM (C=2, kernel=polynomial; 2)',
                              'kNN (k=1, metric=euclid)', 'kNN (k=1, metric=manhattan)',
                              'kNN (k=5, metric=euclid)', 'kNN (k=5, metric=manhattan)',
                              'Aggregated']
        stacking = StackingClassifier(classifiers)
        stacking.fit(X_train, y_train)
        errors = [classifier_error(classifier, X_test, y_test) for classifier in classifiers]
        errors.append(classifier_error(stacking, X_test, y_test))
        confusion_matrices = [confusion_matrix(y_test, [classifier.predict(x)[0] for x in X_test])
                              for classifier in classifiers]
        confusion_matrices.append(confusion_matrix(y_test, [stacking.predict(x)[0] for x in X_test]))
        fig, axes = plt.subplots(3, 3)
        fig.set_size_inches(12, 12)
        with open('B_' + dataset_name + '_confusion_matrices.txt', 'w') as f:
            for i in range(len(confusion_matrices)):
                print(classifiers_labels[i], file=f)
                print(confusion_matrices[i], file=f)
                print(file=f)
        for i in range(len(confusion_matrices)):
            j = i // 3
            k = i % 3
            plot_confusion_matrix(confusion_matrices[i], classifiers_labels[i], axes[j, k])
        plt.savefig('B_' + dataset_name + '_confusion_matrices.png', dpi=100)
        fig = plt.figure(figsize=(12, 12))
        ax = plt.gca()
        accuracies = [1 - e for e in errors]
        x = [i + .5 for i in range(len(classifiers_labels))]
        ax.bar(x, accuracies, width=.5)
        ax.set_xticks(x)
        ax.set_xticklabels(classifiers_labels, rotation=45, fontsize=8)
        ax.set_ylabel('accuracy')
        plt.tight_layout()
        plt.savefig('B_' + dataset_name + '_accuracy.png', dpi=100)
