from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# Elbow method => Choosing number of clusters
def elbowMethod(X, start, end):
    distortions = []
    nums_of_clusters = range(start, end)
    for k in nums_of_clusters:
        k_means = KMeans(init='k-means++', n_clusters=k, n_init=k)
        k_means.fit(X)
        distortions.append(
            sum(np.min(cdist(X, k_means.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

    # Plot the elbow
    plt.plot(nums_of_clusters, distortions, 'bx-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal number of clusters')
    plt.savefig('images/clustering-params/elbowMethod.png')
    plt.show()
