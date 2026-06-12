'''
From the raw cosine similarity and L1 and Euclidean distance data for partial embedding vectors, pick the instances where there is a conflicting change in the categories Delete and Modify. By a conflicting change we mean an instance where the cosince similarity increased or L1 or Euclidean distance decreased between perturbation levels.
'''

import pandas as pd

# add savefile name
wheretosave = "cases_with_conflicting_changes_partial.csv"

# add path to the directory of the result files we want to analyse
results_folder = "results_full_and_partial"
if len(results_folder) > 0:
    results_folder += "\\"

results = []

for md in ["embedding_gemma", "qwen3-embedding-8b"]:
    for trun_type in ["first", "max", "min"]:
        for trun_level in [1, 10, 25, 50, 100]:
            for case in [1,2,3,4,5]:
                for cat in ["delete", "modify"]:
                    for lan in ["english", "finnish", "swedish"]:
                        for metric in ["l1_trun", "euclidean_trun"]:
                            data = pd.read_csv(results_folder + "results_" + md[0:5] + "_" + trun_type + str(trun_level) + ".csv", sep=";")

                            data_values = []
                            to_add_line = md + ";" + trun_type + str(trun_level) + ";C" + str(case) + ";" + cat + ";" + lan

                            for lev in range(1,4):
                                data_values.append(data.loc[(data['file'].str.contains("Case_" + str(case))) & (data['file'].str.contains(cat + "_" + str(lev))) & (data['language']==lan) & (data['type'] == cat),[metric]][metric].values.tolist()[0])
                            
                            if data_values[0] > data_values[1]:
                                results.append(
                                    {
                                    "model": md,
                                    "truncation_type": trun_type + str(trun_level),
                                    "case": "C" + str(case),
                                    "perturbation": cat,
                                    "language": lan,
                                    "metric": metric,
                                    "failure_type": "1->2",
                                    "value1": data_values[0],
                                    "value2": data_values[1],
                                    "value3": data_values[2]
                                    }
                                )
                            if data_values[1] > data_values[2]:
                                results.append(
                                    {
                                    "model": md,
                                    "truncation_type": trun_type + str(trun_level),
                                    "case": "C" + str(case),
                                    "perturbation": cat,
                                    "language": lan,
                                    "metric": metric,
                                    "failure_type": "2->3",
                                    "value1": data_values[0],
                                    "value2": data_values[1],
                                    "value3": data_values[2]
                                    }
                                )

df = pd.DataFrame(results)
df.to_csv(wheretosave, index=False, sep=";")