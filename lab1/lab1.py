# coding=utf-8

import random
import statistics
import matplotlib.pyplot as plt
from math import sqrt

__author__ = 'Michał Ciołczyk'


def generate_set(dim):
    X = []
    for i in range(100):
        elem = []
        for j in range(dim):
            elem.append(random.random())
        X.append(elem)
    return X


def get_coefficient(dim):
    def dist(x, y):
        d = 0
        for i in range(dim):
            d += (x[i] - y[i]) ** 2
        return sqrt(d)

    set = generate_set(dim)
    query = []
    for j in range(dim):
        query.append(random.random())
    distances = []
    for i in range(100):
        distances.append(dist(set[i], query))
    return float(statistics.stdev(distances)) / float(statistics.mean(distances))


if __name__ == "__main__":
    dim = 1
    x = []
    y = []
    y_err = []
    while dim < 10000:
        coefficients = []
        for i in range(4):
            coefficients.append(get_coefficient(dim))
        x.append(dim)
        y.append(statistics.mean(coefficients))
        y_err.append(statistics.stdev(coefficients))
        dim *= 2
    print(x, y, y_err)
    plt.yscale('log')
    plt.xscale('log')
    plt.errorbar(x, y, y_err)
    plt.savefig("plot.png")
    plt.show()
