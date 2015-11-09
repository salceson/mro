# coding=utf-8

from sklearn import datasets, svm
import matplotlib.pyplot as plt

from visualisation_util import plot_areas, plot_2d_classes

__author__ = 'Michał Ciołczyk'


def read_data(only_2_features=True):
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    if only_2_features:
        X = X[:, :2]
    return X, y


if __name__ == '__main__':
    X, y = read_data()
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    svm1 = svm.SVC()
    svm1.fit(X, y)
    ax1.set_title("SVC linear")
    plot_areas(lambda x: svm1.predict(x), 0.1, X, ax1)
    plot_2d_classes(X, y, 'ryb', ax1)
    svm2 = svm.SVC(kernel='poly', degree=2)
    svm2.fit(X, y)
    ax2.set_title("SVC polynomial, deg: 2")
    plot_areas(lambda x: svm2.predict(x), 0.1, X, ax2)
    plot_2d_classes(X, y, 'ryb', ax2)
    svm3 = svm.SVC(kernel='poly', degree=3)
    svm3.fit(X, y)
    ax3.set_title("SVC polynomial, deg: 3")
    plot_areas(lambda x: svm3.predict(x), 0.1, X, ax3)
    plot_2d_classes(X, y, 'ryb', ax3)
    svm4 = svm.SVC(kernel='poly', degree=6)
    svm4.fit(X, y)
    ax4.set_title("SVC polynomial, deg: 6")
    plot_areas(lambda x: svm4.predict(x), 0.1, X, ax4)
    plot_2d_classes(X, y, 'ryb', ax4)
    plt.show()
