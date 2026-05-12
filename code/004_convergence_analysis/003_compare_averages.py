'''
Compute how much the averages for partial embedding vectors differ from the averages for full embedding vectors in each truncation category and truncation level.
'''

import pandas as pd

# name of the file containing the averages with respect to the full vectors and the partial vectors
data_filename = "all_original_data_aves_stdevs.csv"

# read the data:
data = pd.read_csv(data_filename, sep=";")

# savefile name
new_results_filename = "all_comparisons_averages.csv"

results = []

for md in ["qwen3-embedding-8b", "embeddinggemma"]:
    for lan in ["english", "finnish", "swedish"]:
        for first_min_max in ["first", "max", "min"]:
            #Delete, Modify, Paraphrase
            for category in ["D", "M", "P"]:
                for l1_or_euclidean in ["l1_ave", "euclidean_ave"]:
                    full_values = data.loc[(data['model']==md) & (data['language']==lan) & (data['perturbation_type_and_level'].str.contains(category)) & (data['truncation_type'] == "all"),[l1_or_euclidean]][l1_or_euclidean].values.tolist()

                    full_values_scaled = [x / full_values[0] for x in full_values]

                    for trun_level in [1, 10, 25, 50, 100]:
                        partial_values = data.loc[(data['model']==md) & (data['language']==lan) & (data['perturbation_type_and_level'].str.contains(category)) & (data['truncation_type'] == first_min_max + str(trun_level)),[l1_or_euclidean]][l1_or_euclidean].values.tolist()

                        partial_values_scaled = [x / partial_values[0] for x in partial_values]

                        rel_diff1 = str((partial_values_scaled[0] - full_values_scaled[0]) / full_values_scaled[0])
                        rel_diff2 = str((partial_values_scaled[1] - full_values_scaled[1]) / full_values_scaled[1])
                        rel_diff3 = str((partial_values_scaled[2] - full_values_scaled[2]) / full_values_scaled[2])

                        results.append(
                            {
                            "model": md,
                            "language": lan,
                            "perturbation_type": category,
                            "truncation_type": first_min_max + str(trun_level),
                            "average_type": l1_or_euclidean,
                            "relative_difference_lev1": rel_diff1,
                            "relative_difference_lev2": rel_diff2,
                            "relative_difference_lev3": rel_diff3,
                            }
                        )

df = pd.DataFrame(results)
df.to_csv(new_results_filename, index=False, sep=";")