# coding=utf-8

from sklearn.manifold import Isomap
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from paper.utils import load_dataset, plot_embedding

__author__ = 'Michał Ciołczyk'

if __name__ == '__main__':
    X = load_dataset('data')
    ssc = StandardScaler()
    X = ssc.fit_transform(X)
    isomap = Isomap()
    X_transformed = isomap.fit_transform(X)
    plot_embedding(X_transformed, X, 'Isomap')
    plt.savefig('isomap.png')
    X = load_dataset('data', False)
    isomap = Isomap()
    X_transformed = isomap.fit_transform(X)
    plot_embedding(X_transformed, X, 'Isomap - whole dataset')
    plt.savefig('isomap_whole.png')