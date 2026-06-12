'''
Create embedding vectors for closed OpenAI models and compute cosine similarity and L1 and Euclidean distances

IMPORTANT:
Running this script successfully requires a working Azure OpenAI connection set up on Prompt flow on VS Code, and the successful installation of the genaid-promptflow package. You can create this connection with your OpenAI API in the Connections section in Prompt flow. After creating it, open the 'flow.dag.yaml' file in the 'comparison-flow_closed' folder and locate the 'GA_OpenAI_Embedding_Model' node in the flow. For 'connection', choose the name of your Azure OpenAI connection. Then save the flow before running the script (remember to also change the correct case folder, result folder and model names below before running it).
'''

import subprocess

# add path to the case file directory; the path is relative to the 'comparison-flow_closed' directory (i.e., if you just use a directory name instead of the full path, the script assumes that the directory is in the 'comparison-flow_closed' directory)
casefolder = "cases"
if len(casefolder) > 0:
    casefolder += "\\"

# add path to the directory where you want to save the results; the path is relative to the 'comparison-flow_closed' directory (i.e., if you just use a directory name instead of the full path, the script assumes that the directory is in the 'comparison-flow_closed' directory)
savefolder = "results"
if len(savefolder) > 0:
    savefolder += "\\"

# check that the embedding model names correspond to the ones you are using
for mod in ["text-embedding-ada-002-2", "text-embedding-3-large"]:
    wheretosave = ""
    if mod == "text-embedding-ada-002-2":
        wheretosave = savefolder + "results_ada.csv"
    else:
        wheretosave = savefolder + "results_large.csv"
    for lan in ["english", "finnish", "swedish"]:
        for case in range(1,6):
            for perturbation in ["delete", "modify", "paraphrase"]:
                for level in range(1,4):
                    # original summary
                    filenameref_folder = casefolder + lan + "_cases\\medicine_" + str(case) + "\\original\\"
                    filenameref_name = "Case_" + str(case) + "_Medicine_" + lan.capitalize() + "_summary.csv"
                    filenameref = filenameref_folder + filenameref_name

                    # perturbated summary
                    filenamemod_folder = casefolder + lan + "_cases\\medicine_" + str(case) + "\\" + perturbation + "\\"
                    filenamemod_name = "Case_" + str(case) + "_Medicine_" + lan.capitalize() + "_summary_" + perturbation + "_" + str(level) + ".csv"
                    filenamemod = filenamemod_folder + filenamemod_name

                    # construct the command for running the flow
                    command = "pf flow test --flow comparison-flow_closed --inputs "
                    command += "filename_ref=" + filenameref + " "
                    command += "filename_mod=" + filenamemod + " "
                    command += "savefile=" + wheretosave + " "
                    command += "em_model=" + mod + " "
                    command += "language=" + lan + " "
                    command += "case=" + "medicine_" + str(case) + " "
                    command += "type=" + perturbation + " "
                    command += "filename_mod_tosave=" + filenamemod_name

                    subprocess.run(command) 
