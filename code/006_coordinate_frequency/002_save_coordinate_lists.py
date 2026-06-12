'''
Use the saved embedding vectors to generate and save lists of tuples consisting of first, biggest or smallest coordinate changes between the embedding vector of the original summary and the embedding vector of the perturbed summary. The script uses the flow in the 'vector-coordinates' directory which was built using Prompt flow.
'''

import subprocess

# add path to the directory containing the embedding vector files; the path is relative to the 'vector-coordinates' directory (i.e., if you just use a directory name instead of the full path, the script assumes that the directory is in the 'vector-coordinates' directory)
vectorfolder = "embedding_vectors"
if len(vectorfolder) > 0:
    vectorfolder += "\\"

# add path to the directory where you want to save the results; the path is relative to the 'vector-coordinates' directory (i.e., if you just use a directory name instead of the full path, the script assumes that the directory is in the 'vector-coordinates' directory)
savefolder = "coordinate_results"
if len(savefolder) > 0:
    savefolder += "\\"

# choose the parameters for which you want analyse the coordinate frequency; for example, 'min_max_first = "max"' and 'how_many = 100' saves a list of 100 tuples consisting of the biggest coordinate-level changes between the embedding vector of the original summary and its modified version and the corresponding coordinates of these changes
min_max_first = "max"
how_many = 100

for md in ["qwen3-embedding-8b:Q6_K", "embeddinggemma:F32"]:
    for lan in ["english", "finnish", "swedish"]:
        for case in range(1,6):
            for perturbation in ["delete", "modify", "paraphrase"]:
                for level in range(1,4):
                    original_summary = vectorfolder + md[0:5] + "_" + lan + "_C" + str(case) + "_original"
                    perturbed_summary = vectorfolder + md[0:5] + "_" + lan + "_C" + str(case) + perturbation[0].capitalize() + str(level)

                    savefile_name = savefolder + md[0:5] + "_" + lan + "_C" + str(case) + perturbation[0].capitalize() + str(level) + "_" + min_max_first + str(how_many)

                    command = "pf flow test --flow vector-coordinates --inputs "
                    command += "file_original=" + original_summary + " "
                    command += "file_reference=" + perturbed_summary + " "
                    command += "savefile_path=" + savefile_name + " "
                    command += "how_many=" + str(how_many) + " "
                    command += "min_max_first=" + min_max_first

                    subprocess.run(command) 

