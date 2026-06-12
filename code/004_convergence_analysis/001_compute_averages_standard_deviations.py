'''
Compute the L1 and Euclidean distance averages and standard deviations in each language over the different cases in the same combination of embedding model, language, perturbation category, perturbation level, truncation type and truncation level.
'''

import numpy as np
import pandas as pd

# add savefile name
savefilename = "all_original_data_aves_stdevs.csv"

# add path to the directory of the result files you want to analyse; the script assumes that the result files for both the full embedding vectors and truncated embedding vectors are in the same directory
casefolder = "results_full_and_partial"
if len(casefolder) > 0:
    casefolder += "\\"

results = []
for md in ["embeddinggemma", "qwen3-embedding-8b"]:
    data = pd.read_csv(casefolder + "results_" + md[0:5] + ".csv", sep=";")
    for lan in ["english", "finnish", "swedish"]:
        for cat in ["delete", "modify", "paraphrase"]:
            for lev in range(1,4):
                data_lim1 = data.loc[(data['language']==lan) & (data['model']==md) & (data['type'] == cat) & (data['file'].str.contains(str(lev) + ".csv")),['l1_distance']]['l1_distance'].values.tolist()
                data_lim2 = data.loc[(data['language']==lan) & (data['model']==md) & (data['type'] == cat) & (data['file'].str.contains(str(lev) + ".csv")),['euclidean_distance']]['euclidean_distance'].values.tolist()

                results.append(
                    {
                        "model": md,
                        "language": lan,
                        "perturbation_type_and_level": cat[0].capitalize() + str(lev),
                        "truncation_type": "all",
                        "l1_ave": np.average(data_lim1),
                        "euclidean_ave": np.average(data_lim2),
                        "l1_stdev": np.std(data_lim1),
                        "euclidean_stdev": np.std(data_lim2),
                    }
                )


for md in ["embeddinggemma", "qwen3-embedding-8b"]:
    for trun_type in ["first", "max", "min"]:
        for how_many in [1, 10, 25, 50, 100]:
            data = pd.read_csv(casefolder + "results_" + md[0:5] + "_" + trun_type + str(how_many) + ".csv", sep=";")
            for lan in ["english", "finnish", "swedish"]:
                for cat in ["delete", "modify", "paraphrase"]:
                    for lev in range(1,4):
                        data_lim1 = data.loc[(data['language']==lan) & (data['model']==md) & (data['type'] == cat) & (data['file'].str.contains(str(lev) + ".csv")),['l1_trun']]['l1_trun'].values.tolist()
                        data_lim2 = data.loc[(data['language']==lan) & (data['model']==md) & (data['type'] == cat) & (data['file'].str.contains(str(lev) + ".csv")),['euclidean_trun']]['euclidean_trun'].values.tolist()

                        results.append(
                            {
                                "model": md,
                                "language": lan,
                                "perturbation_type_and_level": cat[0].capitalize() + str(lev),
                                "truncation_type": trun_type + str(how_many),
                                "l1_ave": np.average(data_lim1),
                                "euclidean_ave": np.average(data_lim2),
                                "l1_stdev": np.std(data_lim1),
                                "euclidean_stdev": np.std(data_lim2),
                            }
                        )

df = pd.DataFrame(results)
df.to_csv(savefilename, index=False, sep=";")