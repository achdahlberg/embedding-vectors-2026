'''
Create embedding vectors for open models and compute L1 and Euclidean distances for partial embedding vectors

IMPORTANT:
Running this script successfully requires Ollama installed on the device you are using, and the embedding models you want to compare pulled to Ollama. You need to change the model names below to the ones on your computer (run 'ollama list' in your terminal to get the correct names).
'''

import os

import pandas as pd
from langchain_ollama import OllamaEmbeddings

# Import tools for computing the values from the 'tools' folder
from tools.truncated_l1_distance import compute_truncated_l1_distance
from tools.truncated_l2_distance import compute_truncated_l2_distance

# Check that the model names correspond to the ones installed on your device
for OLLAMA_MODEL in ["embeddinggemma:F32", "qwen3-embedding-8b:Q6_K"]:
    for min_max_first in ["min", "max", "first"]:
        for trun_level in [1, 10, 25, 50, 100]:

            # Short model name for savefile
            MODEL_FILETITLE = OLLAMA_MODEL[0:5]

            # Model name for the results in the file; remove the quantization info from the name
            MODEL_NAME = ""
            if OLLAMA_MODEL == "qwen3-embedding-8b:Q6_K":
                MODEL_NAME = "qwen3-embedding-8b"
            else:
                MODEL_NAME = "embeddinggemma"

            # Default Ollama local server
            OLLAMA_URL = "http://127.0.0.1:11434"

            # Directory for cases and path for result file for the model
            BASE_DIR = "cases"
            OUTPUT_PATH = "results_trun\\results_" + MODEL_FILETITLE + "_" + min_max_first + str(trun_level) + ".csv"

            embedding = OllamaEmbeddings(model=OLLAMA_MODEL, base_url=OLLAMA_URL)


            # Utility to load text from a file
            def read_text(file_path):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        return f.read().strip()
                except Exception as e:
                    print(f"Failed to read {file_path}: {e}")
                    return ""

            # Storage
            results = []

            # Loop over languages
            for lang_folder in ["english_cases", "finnish_cases", "swedish_cases"]:
                lang_path = os.path.join(BASE_DIR, lang_folder)
                LANGUAGE = lang_folder.replace("_cases", "")

                for case_dir in os.listdir(lang_path):
                    case_path = os.path.join(lang_path, case_dir)
                    if not os.path.isdir(case_path):
                        continue

                    original_dir = os.path.join(case_path, "original")
                    if not os.path.exists(original_dir):
                        print(f"No 'original' folder in {case_dir}")
                        continue

                    original_files = [f for f in os.listdir(original_dir) if f.endswith(".csv")]
                    if not original_files:
                        print(f"No CSV found in {original_dir}")
                        continue

                    original_file_path = os.path.join(original_dir, original_files[0])
                    original_text = read_text(original_file_path)
                    if not original_text:
                        continue

                    # Process perturbations
                    for variation in ["delete", "modify", "paraphrase"]:
                        variation_path = os.path.join(case_path, variation)
                        if not os.path.exists(variation_path):
                            continue

                        for file in os.listdir(variation_path):
                            if not file.endswith(".csv"):
                                continue

                            mod_file_path = os.path.join(variation_path, file)
                            mod_text = read_text(mod_file_path)
                            if not mod_text:
                                continue

                            try:
                                # Compute the values
                                l1_trun = compute_truncated_l1_distance(embedding, original_text, mod_text, min_max_first, trun_level)
                                euclidean_trun = compute_truncated_l2_distance(embedding, original_text, mod_text, min_max_first, trun_level)

                                results.append(
                                    {
                                        "model": MODEL_NAME,
                                        "case": case_dir,
                                        "language": LANGUAGE,
                                        "type": variation,
                                        "file": file,
                                        "truncation_level": min_max_first + str(trun_level),
                                        "l1_trun": l1_trun,
                                        "euclidean_trun": euclidean_trun,
                                    }
                                )
                                print(f"{case_dir} / {file} processed.")
                            except Exception as e:
                                print(f"Error processing {file}: {e}")

            # Save results
            df = pd.DataFrame(results)
            df.to_csv(OUTPUT_PATH, index=False, sep=";")
            print(f"\n All results saved to: {OUTPUT_PATH}")
