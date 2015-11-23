# coding=utf-8
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from paper.utils import load_dataset, plot_embedding

__author__ = 'Michał Ciołczyk'

if __name__ == '__main__':
    X = load_dataset('data')
    pca = PCA(n_components=2)
    X_transformed = pca.fit_transform(X)
    plot_embedding(X_transformed, X, 'Test')
    plt.show()
