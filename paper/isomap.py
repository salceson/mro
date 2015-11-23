# coding=utf-8

from sklearn.manifold import Isomap
import matplotlib.pyplot as plt

from paper.utils import load_dataset, plot_embedding

__author__ = 'Michał Ciołczyk'

if __name__ == '__main__':
    X = load_dataset('data')
    isomap = Isomap()
    X_transformed = isomap.fit_transform(X)
    plot_embedding(X_transformed, X, 'Test')
    plt.show()
