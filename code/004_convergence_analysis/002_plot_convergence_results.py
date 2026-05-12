'''
Plot the convergence results for the average values and standard deviations we computed in the previous script.
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.colors import to_hex
import pandas as pd

# path for the file containing the data 
results_file = "all_original_data_aves_stdevs.csv"

# folder where we save the plots (add \\ to the end of the folder name)
savefolder = "plots\\"

# function for plot colors
def shade(color, mix_with=(0.9, 0.9, 0.9), alpha=0.0):
    """
    Blend base 'color' with 'mix_with' (a light grey) by factor alpha in [0..1].
    alpha=0: original color, alpha=1: mostly mix_with.
    This gives lighter, slightly desaturated 'matte' versions.
    """
    base = np.array(to_rgb(color))
    mix = np.array(mix_with)
    # pre-desaturate a touch by nudging base toward neutral grey
    grey = np.array([0.7, 0.7, 0.7])
    base = 0.8 * base + 0.2 * grey
    return tuple((1 - alpha) * base + alpha * mix)

for lan in ["english", "finnish", "swedish"]:
    for md in ["qwen3-embedding-8b", "embeddinggemma"]:
        for first_min_max in ["first", "max", "min"]:
            for category in ["D", "M", "P"]:
                cat_name = ""
                if category == "D":
                    cat_name = "delete"
                elif category == "M":
                    cat_name = "modify"
                else:
                    cat_name = "paraphrase"
                for l1_or_euclidean in ["l1", "euclidean"]:
                    # name of the plot file:
                    plot_filename = savefolder + lan[0:3] + "_" + md[0:5] + "_" + first_min_max + "_" + category + "_" + l1_or_euclidean + ".svg"
                    
                    # read the data from the results file:
                    data = pd.read_csv(results_file, sep=";")

                    # empty lists for the L1/Euclidean distance values and standard deviations for the plot:
                    level1 = []
                    level2 = []
                    level3 = []
                    stdevs1 = []
                    stdevs2 = []
                    stdevs3 = []

                    # raw L1/Euclidean values for full vectors
                    to_plot2 = data.loc[(data['model']==md) & (data['language']==lan) & (data['perturbation_type_and_level'].str.contains(category)) & (data['truncation_type'] == "all"),[l1_or_euclidean + "_ave"]][l1_or_euclidean  + "_ave"].values.tolist()
                    # scaled L1/Euclidean values:
                    to_add2 = [x / to_plot2[0] for x in to_plot2]

                    # raw standard deviation values for full vectors
                    to_plot2_stdev = data.loc[(data['model']==md) & (data['language']==lan) & (data['perturbation_type_and_level'].str.contains(category)) & (data['truncation_type'] == "all"),[l1_or_euclidean + "_stdev"]][l1_or_euclidean  + "_stdev"].values.tolist()
                    # scaled standard deviations values (notice that we still scale with respect to the level 1 L1/Euclidean values)
                    to_add2_stdev = [x / to_plot2[0] for x in to_plot2_stdev]

                    # add values to the corresponding lists
                    level1.append(to_add2[0])
                    level2.append(to_add2[1])
                    level3.append(to_add2[2])
                    stdevs1.append(to_add2_stdev[0])
                    stdevs2.append(to_add2_stdev[1])
                    stdevs3.append(to_add2_stdev[2])

                    # repeat the previous process for truncated values
                    for trun_level in [1, 10, 25, 50, 100]:
                        to_plot = data.loc[(data['model']==md) & (data['language']==lan) & (data['perturbation_type_and_level'].str.contains(category)) & (data['truncation_type'] == first_min_max + str(trun_level)),[l1_or_euclidean + "_ave"]][l1_or_euclidean  + "_ave"].values.tolist()
                        to_add = [x / to_plot[0] for x in to_plot]

                        to_plot_stdev = data.loc[(data['model']==md) & (data['language']==lan) & (data['perturbation_type_and_level'].str.contains(category)) & (data['truncation_type'] == first_min_max + str(trun_level)),[l1_or_euclidean + "_stdev"]][l1_or_euclidean  + "_stdev"].values.tolist()
                        to_add_stdev = [x / to_plot[0] for x in to_plot_stdev]

                        level1.append(to_add[0])
                        level2.append(to_add[1])
                        level3.append(to_add[2])
                        stdevs1.append(to_add_stdev[0])
                        stdevs2.append(to_add_stdev[1])
                        stdevs3.append(to_add_stdev[2])

                    # set colors for the bars to match the other figures in the paper
                    color_base = ""
                    colors = []
                    if md == "qwen3-embedding-8b":
                        color_base = "#9467bd"
                    else:
                        color_base = "#d62728"
                    for i in [0.6, 0.3, 0.0]:
                        c = shade(color_base, mix_with=(0.9, 0.9, 0.9), alpha=i)
                        colors.append(c)

                    # start of the plot construction
                    if category == "D":
                        fig, ax = plt.subplots(figsize =(8.5, 4.1))
                    else:
                        fig, ax = plt.subplots(figsize =(8.5, 3.8))
                    barWidth = 0.25
                    offsets = {1: 0.0, 2: barWidth, 3: 2*barWidth}

                    br1 = np.arange(len(level1)) 
                    br2 = [x + barWidth for x in br1] 
                    br3 = [x + barWidth for x in br2]

                    plt.bar(br1, level1, color = to_hex(colors[0]), width = barWidth, yerr = stdevs1, capsize=3, 
                            edgecolor ='black', label = category + '1') 
                    plt.bar(br2, level2, color  = to_hex(colors[1]), width = barWidth, yerr = stdevs2, capsize=3,
                            edgecolor ='black', label = category + '2') 
                    plt.bar(br3, level3, color  = to_hex(colors[2]), width = barWidth, yerr = stdevs3, capsize=3,
                            edgecolor ='black', label = category + '3')

                    #plt.xlabel("truncation level", fontweight ='bold', fontsize = 10) 
                    plt.ylabel(cat_name.capitalize(), fontweight ='bold', fontsize = 10) 
                    plt.xticks([r + barWidth for r in range(len(level1))], 
                            ["all", first_min_max + "1", first_min_max + "10", first_min_max + "25", first_min_max + "50", first_min_max + "100"])
                    #the same scale for everything, adjusted for max:
                    ax.set_ylim(0.5, 3.35)
                    #different scales:
                    #plt.yticks(np.arange(0.5, max(level3) + max(stdevs3) + 0.1, 0.25))
                    plt.yticks([0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25])
                    #plt.yticks(np.arange(0.5, 3.35))

                    # lower the x labels and remove the ticks
                    ax.tick_params(axis='x', which="both", pad=8, bottom=False, top=False)

                    for lvl in [1,2,3]:
                        for i in range(0,6):
                            xpos = i + offsets[lvl]
                            ax.text(
                                xpos,
                                -0.01,
                                str(lvl),
                                transform=ax.get_xaxis_transform(),
                                ha="center",
                                va="top",
                                fontsize=9,
                            )

                    # add dashed lines to the levels of the untruncated values
                    plt.axhline(y=to_add2[0], ls='--', linewidth=1.0, color='black')
                    plt.axhline(y=to_add2[1], ls='--', linewidth=1.0, color='black')
                    plt.axhline(y=to_add2[2], ls='--', linewidth=1.0, color='black')

                    # add title and label to the correct pictures
                    if category == "D" and md == "qwen3-embedding-8b":
                        plt.title("Qwen3-Embedding-8B")
                        ax.text(
                            -0.04,
                            1.04,
                            "a.",
                            transform=ax.transAxes,
                            fontsize=12,
                            fontweight="bold",
                            va="bottom",
                            ha="right",
                        )
                    if category == "D" and md == "embeddinggemma":
                        plt.title("EmbeddingGemma")
                        ax.text(
                            -0.04,
                            1.04,
                            "b.",
                            transform=ax.transAxes,
                            fontsize=12,
                            fontweight="bold",
                            va="bottom",
                            ha="right",
                        )
                    #plt.legend()
                    plt.tight_layout()
                    plt.savefig(plot_filename, dpi=500)
                    plt.close()

