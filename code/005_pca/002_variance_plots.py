'''
Plot the cumulative variance of the embedding vectors when we have ordered the dimensions from biggest variance to smallest variance; we use the PCA results for this.
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re

# add the embedding model name you want to use in the plot
modelname = "EmbeddingGemma"

# add path to the PCA result directory
resultfolder = "PCA_embeddinggemma"
if len(resultfolder) > 0:
    resultfolder += "\\"

# add path to the directory where you want to save the variance plot
savefolder = ""
if len(savefolder) > 0:
    savefolder += "\\"

#define function to calculate Gini coefficient
def gini(x):
    total = 0
    for i, xi in enumerate(x[:-1], 1):
        total += np.sum(np.abs(xi - x[i:]))
    return total / (len(x)**2 * np.mean(x))

variance_file_name = resultfolder + "variance_per_dimension.csv"

data= pd.read_csv(variance_file_name, sep=";")
variances = data.loc[:]['variance'].values.tolist()
variances.sort(reverse=True)

variances_sum = sum(variances)
A = [ 100*(sum(variances[0:i]))/variances_sum for i in range(0,len(variances))]

gini_coef = gini(np.array(variances))

plt.figure(figsize=(7, 4))
plt.plot(np.arange(1, len(A) + 1), A, linewidth=1)  # Variance curve
plt.xlabel("Dimension sorted by variance")
plt.ylabel("Cumulative variance %")
plt.title(f"{modelname} (Gini={gini_coef:.3f})")
plt.tight_layout()
plt.savefig(modelname.lower()[0:5] + "_cumulative_variance_plot.svg", dpi=500)
plt.close()