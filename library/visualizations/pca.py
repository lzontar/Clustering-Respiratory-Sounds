import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def pcaAnalysis(X, y, title, attrs, separatorCol, clusterLabels, filepath=None):
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(X)
    principalDf = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2'])

    loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'],
                            index=attrs)
    print(loadings)
    finalDf = pd.concat([principalDf, y], axis=1)

    fig = plt.figure(figsize=(18, 12))

    plt.xlabel('Principal Component 1 (PC1)', fontsize=26)
    plt.ylabel('Principal Component 2 (PC2)', fontsize=26)
    plt.title(title, fontsize=30)

    targets = list(set(y[separatorCol]))
    palette = sns.color_palette(None, len(targets))

    labels = []
    for target, color in zip(targets, palette):
        indicesToKeep = finalDf[separatorCol] == target
        plt.scatter(finalDf.loc[indicesToKeep, 'PC1'],
                    finalDf.loc[indicesToKeep, 'PC2'],
                    c=np.array(color).reshape(1, -1),
                    s=20)
        labels.append(f'Cluster {target} ({clusterLabels[str(target)]})')
    plt.legend(labels, loc='upper right', fontsize=26, frameon=True, fancybox=True, shadow=True, facecolor='white', framealpha=1)
    if filepath is not None:
        plt.savefig(f'{filepath}.png')
        loadings.to_csv(f'{filepath}.csv', index=True)
    plt.show()
