'''
Generate and save embedding vectors for each case file with respect to the given embedding models. The script uses the flow in the 'embedding-vectors' directory which was built using Prompt flow.
'''

import subprocess

# add path to the case file directory; the path is relative to the 'embedding-vectors' directory (i.e., if you just use a directory name instead of the full path, the script assumes that the directory is in the 'embedding-vectors' directory)
casefolder = "cases"
if len(casefolder) > 0:
    casefolder += "\\"

# add path to the directory where you want to save the results; the path is relative to the 'embedding-vectors' directory (i.e., if you just use a directory name instead of the full path, the script assumes that the directory is in the 'embedding-vectors' directory)
savefolder = "results"
if len(savefolder) > 0:
    savefolder += "\\"

for md in ["qwen3-embedding-8b:Q6_K", "embeddinggemma:F32"]:
    for lan in ["english", "finnish", "swedish"]:
        for case in range(1,6):
            # original summary
            filenameref_folder = casefolder + lan + "_cases\\medicine_" + str(case) + "\\original\\"
            filenameref_name = "Case_" + str(case) + "_Medicine_" + lan.capitalize() + "_summary.csv"
            filenameref = filenameref_folder + filenameref_name

            # construct the command for running the flow
            command = "pf flow test --flow embedding-vectors --inputs "
            command += "model=" + md + " "
            command += "casefile=" + filenameref + " "
            command += "savefile=" + savefolder + md[0:5] + "_" + lan + "_C" + str(case) + "_original"
            subprocess.run(command) 

            for perturbation in ["delete", "modify", "paraphrase"]:
                for level in range(1,4):
                    # perturbated summary
                    filenamemod_folder = casefolder + lan + "_cases\\medicine_" + str(case) + "\\" + perturbation + "\\"
                    filenamemod_name = "Case_" + str(case) + "_Medicine_" + lan.capitalize() + "_summary_" + perturbation + "_" + str(level) + ".csv"
                    filenamemod = filenamemod_folder + filenamemod_name

                    # construct the command for running the flow
                    command = "pf flow test --flow embedding-vectors --inputs "
                    command += "model=" + md + " "
                    command += "casefile=" + filenamemod + " "
                    command += "savefile=" + savefolder + md[0:5] + "_" + lan + "_C" + str(case) + perturbation[0].capitalize() + str(level)
                    subprocess.run(command) 