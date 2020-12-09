from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score

def dbscanParams(X, epsilons, minSamplesList):
    maxMinSamples = None
    maxEps = None
    maxSilh = None
    for eps in epsilons:
        for minSamples in minSamplesList:
            dbscan = DBSCAN(eps=eps, min_samples=minSamples)
            clusters = dbscan.fit(X)
            coreSamplesMask = np.zeros_like(clusters.labels_, dtype=bool)
            coreSamplesMask[clusters.core_sample_indices_] = True
            labels = clusters.labels_
            if len(set(labels)) > 1 and len(set(labels)) < 12:
                silhouetteAvg = silhouette_score(X, labels)
                if maxSilh is None or maxSilh < silhouetteAvg:
                    maxSilh = silhouetteAvg
                    maxMinSamples = minSamples
                    maxEps = eps
    return maxEps, maxMinSamples