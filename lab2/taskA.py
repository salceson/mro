# coding=utf-8

from sklearn.decomposition import PCA, KernelPCA
import numpy as np
from matplotlib import pyplot as plt

__author__ = 'Michał Ciołczyk'


def generate_dataset_circles(n):
    def generate_circle(n1, r_max, r_min, cls):
        n1 = int(n1)
        ts = np.random.uniform(0, 2 * np.pi, n1)
        rs = np.random.uniform(r_min, r_max, n1)
        points = np.array([[rs[i] * np.cos(ts[i]), rs[i] * np.sin(ts[i])] for i in range(n1)])
        ys = [cls for _ in range(n1)]
        return points, ys

    X1, y1 = generate_circle(3 * n / 7, 0, 0.4, 0)
    X2, y2 = generate_circle(4 * n / 7, 0.4, 1, 1)
    return np.concatenate((X1, X2)), np.concatenate((y1, y2)).ravel()


def generate_cross_shape_dataset(n):
    xs = np.random.uniform(-1, 1, n)
    drags = np.random.uniform(-0.1, 0.1, n)
    n_over_2 = int(n/2)
    X1 = np.array([[xs[i], xs[i] + drags[i]] for i in range(n_over_2)])
    X2 = np.array([[xs[i], -xs[i] + drags[i]] for i in range(n_over_2, n)])
    X = np.concatenate((X1, X2))
    y1 = [0 for _ in range(n_over_2)]
    y2 = [1 for _ in range(n_over_2, n)]
    y = np.concatenate((y1, y2)).ravel()
    return X, y


def draw_dataset(X, y, ax, description):
    ax.set_title(description)
    colors = {0: 'r', 1: 'b'}
    for i in range(2):
        idx = np.where(y == i)
        ax.scatter(X[idx, 0], X[idx, 1], c=colors[i])


def process_dataset(X, y, ax, pca=False, kernel=None, gamma=None, description=''):
    if not pca:
        pca_obj = PCA(2)
        draw_dataset(X, y, ax, description)
        pca_obj = pca_obj.fit(X)
        mean = pca_obj.mean_
        components = pca_obj.components_
        variances = pca_obj.explained_variance_ratio_
        vec1 = [mean[i] + components[0][i] for i in range(len(mean))]
        vec2 = [mean[i] + components[1][i] for i in range(len(mean))]
        ax.arrow(mean[0], mean[1], 2 * vec1[0] * variances[0], 2 * vec1[1] * variances[0], fc='y', ec='y')
        ax.arrow(mean[0], mean[1], 2 * vec2[0] * variances[1], 2 * vec2[1] * variances[1], fc='y', ec='y')
    else:
        kwargs = {'gamma': gamma} if gamma else {}
        pca_obj = KernelPCA(2, kernel=kernel, **kwargs) if kernel else PCA(2)
        X_transformed = pca_obj.fit_transform(X)
        draw_dataset(X_transformed, y, ax, description)


if __name__ == '__main__':
    X1, y1 = generate_dataset_circles(500)
    fig, (row1, row2) = plt.subplots(2, 4, figsize=(15, 15), dpi=80)
    (ax1, ax2, ax3, ax4) = row1
    process_dataset(X1, y1, ax1, description='Set 1 (no PCA)')
    process_dataset(X1, y1, ax2, pca=True, description='Set 1 (Linear PCA)')
    process_dataset(X1, y1, ax3, pca=True, kernel='cosine', description='Set 1 (cosine PCA)')
    process_dataset(X1, y1, ax4, pca=True, kernel='rbf', gamma=3, description='Set 1 (rbf PCA)')
    X2, y2 = generate_cross_shape_dataset(500)
    (ax1, ax2, ax3, ax4) = row2
    process_dataset(X2, y2, ax1, description='Set 2 (no PCA)')
    process_dataset(X2, y2, ax2, pca=True, description='Set 2 (Linear PCA)')
    process_dataset(X2, y2, ax3, pca=True, kernel='cosine', description='Set 2 (cosine PCA)')
    process_dataset(X2, y2, ax4, pca=True, kernel='rbf', gamma=3, description='Set 2 (rbf PCA)')
    plt.savefig('taskA.png')
    plt.show()
