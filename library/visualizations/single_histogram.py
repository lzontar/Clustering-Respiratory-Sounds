from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
import math
import statistics
import numpy as np
import seaborn as sns
import matplotlib as mpl

mpl.style.use('seaborn')
sns.set(font_scale=1.4)


def plotSingleHistNominal(X, Y, xLabels, colname, savefile=None):

    yPos = np.arange(len(Y))
    plt.figure(figsize=(9, 9))

    plt.bar(yPos, X, align='center', alpha=0.5)
    plt.xticks(yPos, xLabels, rotation=30)
    plt.xlabel(colname)
    plt.ylabel('Density')
    plt.title(f"{colname} distribution")
    if savefile is not None:
        plt.savefig(savefile)

    plt.show()
