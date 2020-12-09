from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt

# Calinski-Harabasz Criterion for optimal k
def calinskiHarabaszScore(X, start, end):
    n_of_clusters = range(start, end)

    calinski_harabasz_array = []
    for k in n_of_clusters:
        k_means = KMeans(init='k-means++', n_clusters=k, n_init=k)
        k_means_predict = k_means.fit_predict(X)
        tmp = metrics.calinski_harabasz_score(X, k_means_predict)
        calinski_harabasz_array.append(tmp)

    # Plot the CH Indices
    plt.plot(n_of_clusters, calinski_harabasz_array, 'bx-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Calinski-Harabasz Criterion')
    plt.title('Calinski-Harabasz Criterion for choosing optimal k')
    plt.savefig('images/clustering-params/calinskiHarabaszScore.png')
    plt.show()
