'''
From the raw cosine similarity and L1 and Euclidean distance data, pick the instances where there is a conflicting change in the categories Delete and Modify. By a conflicting change we mean an instance where the cosince similarity increased or L1 or Euclidean distance decreased between perturbation levels (which is not what we want).
'''

import pandas as pd

# add path to the directory of the result files we want to analyse; the script assumes that all the results are in the same directory
results_folder = "results_all"
if len(results_folder) > 0:
    results_folder += "\\"

# add savefile name
wheretosave = "cases_with_conflicting_changes.csv"

results = []

# md referst to the short model names used for the result files; change them if you used different models
for md in ["bge-m", "snowf", "embed", "multi", "nomic", "qwen3", "ada", "large"]:
    for case in [1,2,3,4,5]:
        for category in ["delete", "modify"]:
            for lan in ["english", "finnish", "swedish"]:
                for metric in ["cosine_similarity", "euclidean_distance", "l1_distance"]:
                    # read the results from the correct file
                    data = pd.read_csv(results_folder + "results_" + md + ".csv", sep=";")

                    model_name = data['model'].values.tolist()[0]

                    to_add_line = model_name + ";C" + str(case) + ";" + category + ";" + lan

                    # extract the 3 values for different perturbation levels for the corresponding case, category, language and metric
                    data_values = []
                    for lev in range(1,4):
                        data_values.append(data.loc[(data['file'].str.contains("Case_" + str(case))) & (data['file'].str.contains(category + "_" + str(lev))) & (data['language']==lan) & (data['type'] == category),[metric]][metric].values.tolist()[0])
                    
                    # analyse the values: record the instances where cosine similarity increases or other values decrease
                    if metric == "cosine_similarity":
                        if data_values[0] < data_values[1]:
                            results.append(
                                {
                                "model": model_name,
                                "case": "C" + str(case),
                                "perturbation": category,
                                "language": lan,
                                "metric": metric,
                                "failure_type": "1->2",
                                "value1": data_values[0],
                                "value2": data_values[1],
                                "value3": data_values[2]
                                }
                            )
                        if data_values[1] < data_values[2]:
                            results.append(
                                {
                                "model": model_name,
                                "case": "C" + str(case),
                                "perturbation": category,
                                "language": lan,
                                "metric": metric,
                                "failure_type": "2->3",
                                "value1": data_values[0],
                                "value2": data_values[1],
                                "value3": data_values[2]
                                }
                            )
                    else:
                        if data_values[0] > data_values[1]:
                            results.append(
                                {
                                "model": model_name,
                                "case": "C" + str(case),
                                "perturbation": category,
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
                                "model": model_name,
                                "case": "C" + str(case),
                                "perturbation": category,
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
