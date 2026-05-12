'''
Compute and save the coordinate frequency results from the saved coordinate lists.
'''

import pickle

# add path to the directory containing the coordinate result files
coordinatefolder = "coordinate_results"
if len(coordinatefolder) > 0:
    coordinatefolder += "\\"

# add path to the directory where you want to save the results; the results will be saved in txt files containing results of the analysis with respect to different categories
savefolder = "coordinate_frequency_results"
if len(savefolder) > 0:
    savefolder += "\\"

# add the parameters you used for the coordinate results
how_many = 100
min_max_first = "max"

for model in ["qwen3-embedding-8b:Q6_K", "embeddinggemma:F32"]:
    dims = 0
    if model == "qwen3-embedding-8b:Q6_K":
        shortmodel = "qwen3"
        dims = 4096
    elif model == "embeddinggemma:F32":
        shortmodel = "embed"
        dims = 768
    coords_all = list(range(0,dims))
    coords_all_times = [0] * dims
    coords_all_D_times = [0] * dims
    coords_all_M_times = [0] * dims
    coords_all_P_times = [0] * dims

    textfile_all = shortmodel + "_" + "all_" + min_max_first + str(how_many) + "_results.txt"
    textfile_all_D = shortmodel + "_" + "all_delete_" + min_max_first + str(how_many) + "_results.txt"
    textfile_all_M = shortmodel + "_" + "all_modify_" + min_max_first + str(how_many) + "_results.txt"
    textfile_all_P = shortmodel + "_" + "all_paraphrase_" + min_max_first + str(how_many) + "_results.txt"

    for lan in ["english", "finnish", "swedish"]:
        coords_lan_times = [0] * dims

        textfile_lan = shortmodel + "_" + lan + "_" + min_max_first + str(how_many) + "_results.txt"

        for A in ["D", "M", "P"]:
            coords_lan_cat_times = [0] * dims

            category = ""
            if A == "D":
                category = "delete"
            if A == "M":
                category = "modify"
            if A == "P":
                category = "paraphrase"

            textfile_lan_cat = shortmodel + "_" + lan + "_" + category + "_" + min_max_first + str(how_many) + "_results.txt"
            
            for i in range(1,6):
                for j in range(1,4):
                    coordinatefile = coordinatefolder + shortmodel + "_" + lan + "_C" + str(i) + A + str(j) + "_" + min_max_first + str(how_many)

                    vector = []
                    with open(coordinatefile, "rb") as fp:
                        vector = pickle.load(fp)

                    for a in vector:
                        b = a[1]
                        coords_all_times[b] += 1
                        if A == "D":
                            coords_all_D_times[b] += 1
                        if A == "M":
                            coords_all_M_times[b] += 1
                        if A == "P":
                            coords_all_P_times[b] += 1
                        coords_lan_times[b] += 1
                        coords_lan_cat_times[b] += 1

            coords_lan_cat_frequencies = list(zip(coords_all,coords_lan_cat_times))

            sorte = sorted(coords_lan_cat_frequencies, key=lambda tup: tup[1], reverse=True)

            with open(savefolder + textfile_lan_cat, 'w', encoding='utf-8') as f:
                f.write(model + ", " + lan + ", " + category + ", " + min_max_first + str(how_many) + "\n\n")
                f.write("dimensions: " + str(dims) + "\n")
                f.write("difference vectors in this category: 15\n\n")
                f.write("maximum coordinate frequency (out of 15): " + str(max(coords_lan_cat_times)) + "\n")
                f.write("number of zero frequencies (out of " + str(dims) + "): " + str(coords_lan_cat_times.count(0)) + "\n")
                f.write("percentage of zero frequencies: " + str(100 * coords_lan_cat_times.count(0) / dims) + "%\n\n")
                f.write("top 10 maximum frequencies and the corresponding mathematical dimensions:\n")
                for i in range(0,10):
                    f.write("  dim: " + str(sorte[i][0] + 1) + ",   freq: " + str(sorte[i][1]) + "\n")

        coords_lan_frequencies = list(zip(coords_all,coords_lan_times))

        sorte = sorted(coords_lan_frequencies, key=lambda tup: tup[1], reverse=True)

        with open(savefolder + textfile_lan, 'w', encoding='utf-8') as f:
            f.write(model + ", " + lan + ", " + min_max_first + str(how_many) + "\n\n")
            f.write("dimensions: " + str(dims) + "\n")
            f.write("difference vectors in this category: 45\n\n")
            f.write("maximum coordinate frequency (out of 45): " + str(max(coords_lan_times)) + "\n")
            f.write("number of zero frequencies (out of " + str(dims) + "): " + str(coords_lan_times.count(0)) + "\n")
            f.write("percentage of zero frequencies: " + str(100 * coords_lan_times.count(0) / dims) + "%\n\n")
            f.write("top 10 maximum frequencies and the corresponding mathematical dimensions:\n")
            for i in range(0,10):
                f.write("  dim: " + str(sorte[i][0] + 1) + ",   freq: " + str(sorte[i][1]) + "\n")

    coords_all_D_frequencies = list(zip(coords_all,coords_all_D_times))

    sorte = sorted(coords_all_D_frequencies, key=lambda tup: tup[1], reverse=True)

    with open(savefolder + textfile_all_D, 'w', encoding='utf-8') as f:
        f.write(model + ", " + "delete" + ", " + min_max_first + str(how_many) + "\n\n")
        f.write("dimensions: " + str(dims) + "\n")
        f.write("difference vectors in this category: 45\n\n")
        f.write("maximum coordinate frequency (out of 45): " + str(max(coords_all_D_times)) + "\n")
        f.write("number of zero frequencies (out of " + str(dims) + "): " + str(coords_all_D_times.count(0)) + "\n")
        f.write("percentage of zero frequencies: " + str(100 * coords_all_D_times.count(0) / dims) + "%\n\n")
        f.write("top 10 maximum frequencies and the corresponding mathematical dimensions:\n")
        for i in range(0,10):
            f.write("  dim: " + str(sorte[i][0] + 1) + ",   freq: " + str(sorte[i][1]) + "\n")


    coords_all_M_frequencies = list(zip(coords_all,coords_all_M_times))

    sorte = sorted(coords_all_M_frequencies, key=lambda tup: tup[1], reverse=True)

    with open(savefolder + textfile_all_M, 'w', encoding='utf-8') as f:
        f.write(model + ", " + "modify" + ", " + min_max_first + str(how_many) + "\n\n")
        f.write("dimensions: " + str(dims) + "\n")
        f.write("difference vectors in this category: 45\n\n")
        f.write("maximum coordinate frequency (out of 45): " + str(max(coords_all_M_times)) + "\n")
        f.write("number of zero frequencies (out of " + str(dims) + "): " + str(coords_all_M_times.count(0)) + "\n")
        f.write("percentage of zero frequencies: " + str(100 * coords_all_M_times.count(0) / dims) + "%\n\n")
        f.write("top 10 maximum frequencies and the corresponding mathematical dimensions:\n")
        for i in range(0,10):
            f.write("  dim: " + str(sorte[i][0] + 1) + ",   freq: " + str(sorte[i][1]) + "\n")


    coords_all_P_frequencies = list(zip(coords_all,coords_all_P_times))

    sorte = sorted(coords_all_P_frequencies, key=lambda tup: tup[1], reverse=True)

    with open(savefolder + textfile_all_P, 'w', encoding='utf-8') as f:
        f.write(model + ", " + "paraphrase" + ", " + min_max_first + str(how_many) + "\n\n")
        f.write("dimensions: " + str(dims) + "\n")
        f.write("difference vectors in this category: 45\n\n")
        f.write("maximum coordinate frequency (out of 45): " + str(max(coords_all_P_times)) + "\n")
        f.write("number of zero frequencies (out of " + str(dims) + "): " + str(coords_all_P_times.count(0)) + "\n")
        f.write("percentage of zero frequencies: " + str(100 * coords_all_P_times.count(0) / dims) + "%\n\n")
        f.write("top 10 maximum frequencies and the corresponding mathematical dimensions:\n")
        for i in range(0,10):
            f.write("  dim: " + str(sorte[i][0] + 1) + ",   freq: " + str(sorte[i][1]) + "\n")


    coords_all_frequencies = list(zip(coords_all,coords_all_times))

    sorte = sorted(coords_all_frequencies, key=lambda tup: tup[1], reverse=True)

    with open(savefolder + textfile_all, 'w', encoding='utf-8') as f:
        f.write(model + ", " + "all vectors" + ", " + min_max_first + str(how_many) + "\n\n")
        f.write("dimensions: " + str(dims) + "\n")
        f.write("difference vectors in this category: 135\n\n")
        f.write("maximum coordinate frequency (out of 135): " + str(max(coords_all_times)) + "\n")
        f.write("number of zero frequencies (out of " + str(dims) + "): " + str(coords_all_times.count(0)) + "\n")
        f.write("percentage of zero frequencies: " + str(100 * coords_all_times.count(0) / dims) + "%\n\n")
        f.write("top 10 maximum frequencies and the corresponding mathematical dimensions:\n")
        for i in range(0,10):
            f.write("  dim: " + str(sorte[i][0] + 1) + ",   freq: " + str(sorte[i][1]) + "\n")

