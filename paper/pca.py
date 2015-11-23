# coding=utf-8
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from paper.utils import load_dataset, plot_embedding

__author__ = 'Michał Ciołczyk'

if __name__ == '__main__':
    X = load_dataset('data')
    pca = PCA(n_components=2)
    X_transformed = pca.fit_transform(X)
    plot_embedding(X_transformed, X, 'PCA')
    plt.savefig('pca.png')
    ssc = StandardScaler()
    X = ssc.fit_transform(X)
    pca_kernel = KernelPCA(n_components=2, kernel='rbf')
    X_kernel = pca_kernel.fit_transform(X)
    plot_embedding(X_kernel, X, 'Kernel PCA (rbf)')
    plt.savefig('kernel_pca.png')
