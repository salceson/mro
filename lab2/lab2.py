# coding=utf-8
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

__author__ = 'Michał Ciołczyk'


def gen_dataset(n):
    return np.random.multivariate_normal(np.array([0.2, 0.4]), np.array([[3.0, 1.0], [1.0, 2.5]]), n)


if __name__ == '__main__':
    X = gen_dataset(1000)
    pca = PCA(2).fit(X)
    components = pca.components_
    mean = pca.mean_
    variance_ratio = pca.explained_variance_ratio_
    Y = pca.transform(X)
    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.scatter(X[:, 0], X[:, 1])
    vec1 = [mean[i] + components[i, 0] for i in range(len(mean))]
    vec2 = [mean[i] + components[i, 1] for i in range(len(mean))]
    ax1.arrow(mean[0], mean[1], 3 * vec1[0] * variance_ratio[0], 3 * vec1[1] * variance_ratio[0],
              head_width=0.3, head_length=0.8, fc='r', ec='r')
    ax1.arrow(mean[0], mean[1], 3 * vec2[0] * variance_ratio[1], 3 * vec2[1] * variance_ratio[1],
              head_width=0.3, head_length=0.8, fc='r', ec='r')
    ax2.scatter(Y[:, 0], Y[:, 1])
    plt.show()
