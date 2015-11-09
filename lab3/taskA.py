# coding=utf-8

import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import euclidean

from visualisation_util import plot_2d_classes

__author__ = 'Michał Ciołczyk'

EPSILON = 1e-13


def generate_dataset(n=50):
    def generate_points_in_circle(n1, x, y, r, cls):
        ts = np.random.uniform(0, 2 * np.pi, n1)
        rs = np.random.uniform(0, r, n1)
        points = np.array([[rs[i] * np.cos(ts[i]) + x, rs[i] * np.sin(ts[i]) + y] for i in range(n1)])
        ys = [cls for _ in range(n1)]
        return points, ys

    centers_xs = [-3, 0, 3]
    centers_ys = [-3, 0, 3]
    r = 1
    Xs = []
    ys = []
    for i in range(len(centers_xs)):
        x_center = centers_xs[i]
        for j in range(len(centers_ys)):
            y_center = centers_ys[j]
            X_i, y_i = generate_points_in_circle(n, x_center, y_center, r, i * len(centers_ys) + j)
            Xs.append(X_i)
            ys.append(y_i)
    X = Xs[0]
    y = ys[0]
    for i in range(1, len(Xs)):
        X = np.concatenate((X, Xs[i]))
        y = np.concatenate((y, ys[i])).ravel()
    return X, y


def calculate_quality(means, X):
    return np.sum([np.min([euclidean(x, mean) for mean in means]) for x in X])


def k_means(init_method, X, k, iterations):
    means = init_method(X, k)
    # quality = calculate_quality(means, X)
    # while True:
    for i in range(iterations):
        assignment = [[x, np.argmin([euclidean(x, mean) for mean in means])] for x in X]
        new_means = []
        for j in range(len(means)):
            cluster = [x[0] for x in assignment if x[1] == j]
            new_mean_for_cluster = np.average(cluster, axis=0) if len(cluster) > 0 else means[j]
            new_means.append(new_mean_for_cluster)
        new_quality = calculate_quality(new_means, X)
        # if new_quality - quality > -EPSILON:
        #     break
        # quality = new_quality
        means = new_means
        yield (i, new_quality)


def random_init(X, k):
    min_x = np.min(X[:, 0])
    max_x = np.max(X[:, 0])
    min_y = np.min(X[:, 1])
    max_y = np.max(X[:, 1])
    xs = np.random.uniform(min_x, max_x, k)
    ys = np.random.uniform(min_y, max_y, k)
    return np.array([[xs[i], ys[i]] for i in range(k)])


def forgy_init(X, k):
    indices = np.random.random_integers(0, len(X) - 1, k)
    return np.array([X[i] for i in indices])


def random_partition_init(X, k):
    clusters = [[] for _ in range(k)]
    for x in X:
        rand = np.random.random_integers(0, k - 1)
        clusters[rand].append(x)
    return np.array([np.average(clusters[i], axis=0) for i in range(k)])


def kmeanspp_init(X, k):
    means = []
    rand_index = np.random.random_integers(0, len(X) - 1)
    means.append(X[rand_index])
    for x in range(1, k):
        min_distances = [np.min([euclidean(x, mean) for mean in means]) ** 2 for x in X]
        probs = [min_distances[i] / sum(min_distances) for i in range(len(min_distances))]
        new_mean_index = np.random.choice([x for x in range(len(X))], p=probs)
        means.append(X[new_mean_index])
    return np.array(means)


if __name__ == '__main__':
    X, y = generate_dataset(100)
    colors = ['r', 'g', 'b', 'w', 'c', 'm', 'y', 'k', '0.75']
    plot_2d_classes(X, y, colors)
    plt.gca().set_title("Dataset")
    plt.savefig('partA/dataset.png')
    fig = plt.figure()
    ax = plt.gca()
    iterations = 50
    k = 9
    reps = 5
    for method in [random_init, forgy_init, random_partition_init, kmeanspp_init]:
        print(method)
        qualities = []
        for _ in range(reps):
            for i, quality in k_means(method, X, k, iterations):
                qualities.append((i, quality))
        avgs = []
        stds = []
        array = np.array(qualities).reshape(reps, iterations, 2)  # 2 is dim of samples
        for i in range(iterations):
            avgs.append(np.average(array[:, i, 1]))
            stds.append(np.std(array[:, i, 1]))
        ax.errorbar(range(iterations), avgs, yerr=stds, label=method.__name__[:-5])
    ax.legend()
    plt.savefig('partA/qualities.png')
