import json
import time
from collections import Counter

from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn import preprocessing, metrics
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.mixture import GaussianMixture

from library.visualizations.clustering import calinsky_harabasz_score, elbow_method, silhouette_analysis, gapStatistics, dbscanParams
from library.visualizations import pca, single_histogram

import library.file_reader as file_reader
import library.mapper as mapper

import matplotlib.pyplot as plt
import math
import warnings

'''
##################################
SETTINGS
##################################
'''
warnings.filterwarnings('ignore', category=UserWarning)

'''
##################################
DATA READING
##################################
'''
# Read data
# PATIENT_DATA = file_reader.read_data('./data/')
# # Save data in JSON file
# with open('results/data.json', 'w+') as fp:
#     json.dump(PATIENT_DATA, fp)

'''
##################################
DATA PREPROCESSING
##################################
'''
def preprocess(df):
    # Non-clustering features
    nonClusterFeat = list(filter(lambda x: x in ['DIAGNOSIS', 'PATIENT_ID', 'REC_EQUIPMENT', 'CHEST_LOC', 'ACQUISITION', 'HEALTHY', 'SOUND', 'SEX'], df.columns))
    # Clustering features
    clusterFeat = list(filter(lambda x: x not in nonClusterFeat, df.columns))

    # Drop rows with NaN values
    df = df.dropna()

    # Take into account recording data
    df = df[df['REC_EQUIPMENT'] == 3]
    df = df[df['ACQUISITION'] == 0]

    min_max_scaler = preprocessing.MinMaxScaler()
    X_scaled = min_max_scaler.fit_transform(df[clusterFeat].values)
    df_norm = pd.DataFrame(X_scaled, columns=df[clusterFeat].columns)

    return df_norm, df[nonClusterFeat]


'''
##################################
LOAD DATA
##################################
'''
# Load data from JSON file
with open('results/data.json') as json_file:
    PATIENT_DATA = json.load(json_file)

CYCLES_DATA = []
for patient_id in PATIENT_DATA.keys():
    for recording in PATIENT_DATA[patient_id]['RECORDINGS']:
        for cycle_data in recording['RESPIRATORY_CYCLES']:
            cycle = dict()

            cycle['PATIENT_ID'] = patient_id

            cycle['DIAGNOSIS'] = PATIENT_DATA[patient_id]['DIAGNOSIS']
            cycle['HEALTHY'] = 0 if PATIENT_DATA[patient_id]['DIAGNOSIS'] == 6 else 1
            cycle['SOUND'] = 2 if cycle_data['WHEEZES'] == '1' else (1 if cycle_data['CRACKLES'] == '1' else 0)

            cycle['REC_EQUIPMENT'] = recording['REC_EQUIPMENT']
            cycle['CHEST_LOC'] = recording['CHEST_LOC']
            cycle['ACQUISITION'] = recording['ACQUISITION']

            cycle['AGE'] = PATIENT_DATA[patient_id]['AGE']
            cycle['BMI'] = PATIENT_DATA[patient_id]['BMI']
            cycle['SEX'] = PATIENT_DATA[patient_id]['SEX']

            for feature in cycle_data['DATA'].keys():
                ix = 0
                for val in cycle_data['DATA'][feature]:
                    cycle[f"{feature}_{ix}"] = val
                    ix += 1

            CYCLES_DATA.append(cycle)

X = pd.DataFrame(CYCLES_DATA)

# Standardize data frame values to prevent that distances of some values would be interpreted as more important
X_preprocessed, X_ids = preprocess(X)

'''
##################################
VISUALIZATION
##################################
'''
mode = lambda x: x.mode() if len(x) > 2 else np.array(x)
patientDiag = list(X_ids['DIAGNOSIS'])
patientDiagFreqs = list(np.zeros((7,), dtype=int))
for diag in patientDiag:
    patientDiagFreqs[diag] += 1
single_histogram.plotSingleHistNominal(patientDiagFreqs,
                                       list(range(7)),
                                        list(mapper.map_diagnosis.keys())[:7],
                                       'Diagnosis', f"images/diagnosis-distribution.png")

mode = lambda x: x.mode() if len(x) > 2 else np.array(x)
patientDiag = list(X_ids['HEALTHY'])
patientDiagFreqs = list(np.zeros((2,), dtype=int))
for diag in patientDiag:
    patientDiagFreqs[diag] += 1
single_histogram.plotSingleHistNominal(patientDiagFreqs,
                                       list(range(2)),
                                        ['Not healthy', 'Healthy'],
                                       'Healthy', f"images/health-distribution.png")

mode = lambda x: x.mode() if len(x) > 2 else np.array(x)
patientDiag = list(X_ids['SOUND'])
patientDiagFreqs = list(np.zeros((3,), dtype=int))
for diag in patientDiag:
    patientDiagFreqs[diag] += 1
single_histogram.plotSingleHistNominal(patientDiagFreqs,
                                       list(range(3)),
                                        ['Normal sound', 'Crackles', 'Wheezes'],
                                       'Sound', f"images/sound-distribution.png")

'''
##################################
PARAMETERS RETRIEVAL
##################################
'''
# Choosing parameters for algorithms
# calinsky_harabasz_score.calinskiHarabaszScore(X_preprocessed, 2, 10)
# elbow_method.elbowMethod(X_preprocessed, 2, 10)
# silhouette_analysis.silhouetteAnalysis(X_preprocessed, 2, 10)
# gapStatistics.gapStatistics(X_preprocessed)

'''
##################################
FEATURE SUBSETS
##################################
'''
featureSubsets = [
    list(filter(lambda x: 'MFCC' in x, list(X_preprocessed.keys()))),
    list(filter(lambda x: 'ZERO_CROSSING_RATE' in x or 'RMS' in x, list(X_preprocessed.keys()))),
    ['AGE', 'BMI']
]

featureSubsets.append(list(filter(lambda x: x not in featureSubsets[0] and x not in featureSubsets[1] and x not in featureSubsets[2], list(X_preprocessed.keys()))))

'''
##################################
EXECUTION
##################################
'''
nClusterList = [2, 3, 7]
classList = ['HEALTHY', 'SOUND', 'DIAGNOSIS']
labelList = [['Not healthy', 'Healthy'], ['Normal sound', 'Crackles', 'Wheezes'], ['URTI', 'COPD', 'Bronchiectasis', 'Pneumonia', 'Bronchiolitis', 'LRTI', 'Healthy', 'Asthma']]
scenarioName = ['Cepstral domain features', 'Time domain features', 'Demographic features', 'Frequency domain features']

for N_CLUSTERS, CLASS, LABELS in zip(nClusterList, classList, labelList):
    '''
    ##################################
    SAMPLING
    ##################################
    '''
    # We oversample cases for easier interpretation
    smote = SMOTE()
    X_resampled, classSampled = smote.fit_resample(X_preprocessed, X_ids[CLASS])

    ixScenario = 0
    for featureSubset in featureSubsets:
        X_final = X_resampled[featureSubset]
        clusterPredict = {}

        # DBSCAN optimal parameters based on Silhouette score
        eps, minSamples = dbscanParams.dbscanParams(X_final, [0.1, 0.2, 0.3, 0.4, 0.5], list(range(10, 40, 5)))

        '''
        ##################################
        ALGORITHMS INITIALIZATION
        ##################################
        '''
        # Initialization of algorithms
        # K-Means
        k_means = KMeans(init='k-means++', n_clusters=N_CLUSTERS)
        # Expectation–Maximization (EM) Clustering using Gaussian Mixture Models (GMM)
        gmm = GaussianMixture(n_components=N_CLUSTERS)
        # Hierarchical Clustering
        hierarchical_clustering = AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage='ward')
        # Spectral Clustering
        spectral_clustering = SpectralClustering(n_clusters=N_CLUSTERS)

        # Create a list of algorithms
        algorithms = [('K-Means', k_means), ('Gaussian Mixture Model', gmm),
                      ('Hierarchical clustering', hierarchical_clustering), ('Spectral clustering', spectral_clustering)]

        if eps is not None and minSamples is not None:
            # DBSCAN
            dbscan = DBSCAN(eps, minSamples)
            algorithms.append(('DBSCAN', dbscan))

        # Algorithm execution
        for name, alg in algorithms:
            filenamePrefix = f'c{N_CLUSTERS}-sc{ixScenario}-{name}'
            resultsFile = open(f'images/results/scenario{ixScenario}/results/{filenamePrefix}-results.txt', 'w+')

            # Execution of prediction
            print('----- Executing ' + name + ' ----- ')
            resultsFile.write('----- Executing ' + name + ' ----- \n')
            t = time.time()
            clusterPredict[name] = alg.fit_predict(X_final)
            tiempo = time.time() - t
            print(": {:.2f} seconds ".format(tiempo))
            resultsFile.write(": {:.2f} seconds \n".format(tiempo))
            '''
            ##################################
            SCORES
            ##################################
            '''
            # Calinski-Harabasz score
            metric_CH = metrics.calinski_harabasz_score(X_final, clusterPredict[name])
            print("Calinski-Harabasz Index: {:.3f}".format(metric_CH))
            resultsFile.write("Calinski-Harabasz Index: {:.3f}\n".format(metric_CH))

            # Silhouette coefficient
            sample_silhouette = 0.2 if (len(X) > 10000) else 1.0
            metric_SC = metrics.silhouette_score(X_final, clusterPredict[name], metric='euclidean', sample_size=math.floor(sample_silhouette*len(X)), random_state=123456)
            print("Silhouette Coefficient: {:.5f}".format(metric_SC))
            resultsFile.write("Silhouette Coefficient: {:.5f}\n".format(metric_SC))

            nClusters = len(set(clusterPredict[name]))
            '''
            ##################################
            CLUSTER-DIAGNOSIS DISTRIBUTION HEATMAP
            ##################################
            '''
            # Cluster sizes
            print("Cluster sizes: ")
            resultsFile.write("Cluster sizes: \n")
            clusters = pd.DataFrame(clusterPredict[name], index=X_final.index, columns=['cluster'])
            size = clusters['cluster'].value_counts()

            clusterLabelsWSizes = {}
            for num, i in size.iteritems():
                print('%s: %5d (%5.2f%%)' % (num, i, 100 * i / len(clusters)))
                clusterLabelsWSizes[str(num)] = f'{round(100 * i / len(clusters), 2)}%'
                resultsFile.write('%s: %5d (%5.2f%%)\n' % (num, i, 100 * i / len(clusters)))

            dfResults = pd.DataFrame({'CLUSTER': clusterPredict[name], 'CLASS': list(classSampled)})
            clusterCounts = np.zeros([len(set(clusterPredict[name])), len(LABELS)])

            for (ix, row) in dfResults.iterrows():
                clusterCounts[row['CLUSTER']][row['CLASS']] += 1

            for i in range(len(clusterCounts)):
                clusterSum = sum(clusterCounts[i])
                for j in range(len(clusterCounts[i])):
                    clusterCounts[i][j] = clusterCounts[i][j] / clusterSum

            clusterLabels = [f"Cluster {i} ({clusterLabelsWSizes[str(i)]})" for i in set(clusterPredict[name])]
            plt.figure(figsize=(24, 12))
            ax = sns.heatmap(clusterCounts, linewidth=0.5, vmin=0, vmax=1)
            ax.set_xticklabels(LABELS, rotation=30, fontsize=26)
            ax.set_yticklabels(clusterLabels, rotation=0, fontsize=26)
            plt.title(f"Diagnosis distributions over clusters ({name}) - {scenarioName[ixScenario]}", fontsize=32)
            plt.savefig(f'images/results/scenario{ixScenario}/cluster-heatmap/{filenamePrefix}-cluster-heatmap.png')
            plt.savefig(f'../IEEEtran/images/{filenamePrefix}-cluster-heatmap.png')
            plt.show()

            '''
            ##################################
            PCA ANALYSIS
            ##################################
            '''
            X_clusters = pd.DataFrame({'CLUSTER': list(clusterPredict[name])})
            pca.pcaAnalysis(X_final, X_clusters, f'PCA Analysis - {name} - {scenarioName[ixScenario]}', featureSubset, 'CLUSTER', clusterLabelsWSizes, f'images/results/scenario{ixScenario}/pca/{filenamePrefix}-pca')
            pca.pcaAnalysis(X_final, X_clusters, f'PCA Analysis - {name} - {scenarioName[ixScenario]}', featureSubset, 'CLUSTER', clusterLabelsWSizes, f'../IEEEtran/images/{filenamePrefix}-pca')
            resultsFile.close()
        ixScenario += 1
