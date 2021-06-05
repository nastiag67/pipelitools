import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def test_clustering():
    print('test_clustering: ok')


#######################################################################################################################
# K means
#######################################################################################################################

from sklearn.metrics import calinski_harabasz_score


def n_clusters(Z, Kmax=10, n_init = 100):
    """
    if the number of features is larger than the number of samples, then you will be dealing with
    the “curse of dimensionality”, and your k-means algorithm will not produce good results.
    In this case, you do want to reduce the number of features that you have.

    Z - projections on the PCs
        pca = PCA(n_components = ncomp, whiten=True)
        pca.fit(X) #  estimate the parameters of the PCA
        Z = pca.transform(X)

    n_init - Number of different initialisations of the Kmeans algorithm

    Returns: optimal number of clusters where delta is 5 or less
    """
    W_all = np.zeros(shape=(Kmax - 1))
    K_all = np.arange(2, Kmax + 1)
    CH = np.zeros(shape=(Kmax-1))
    for (i, K) in enumerate(K_all):
        kmeans_model = KMeans(n_clusters=K, init='random', n_init=n_init)
        # run the Kmeans algorithm on the PCA variables
        kmeans_model.fit(Z)
        W_all[i] = kmeans_model.inertia_

        labels = kmeans_model.labels_
        CH[i] = calinski_harabasz_score(Z, labels)

        if W_all[i - 1] - W_all[i] <= 15 and W_all[i - 1] != 0:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))
            ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax1.plot(K_all[:i + 1], W_all[:i + 1])
            ax1.set(xlabel='Number of Clusters K', ylabel='W')

            ax2.plot(K_all[:i + 1], CH[:i + 1])
            ax2.set(xlabel='Number of Clusters K', ylabel='Calinski-Harabasz Score')
            return K


if __name__ == '__main__':
    test_clustering()