'''
Create embedding vectors for open models and compute cosine similarity and L1 and Euclidean distances

IMPORTANT:
Running this script successfully requires Ollama installed on the device you are using, and the embedding models you want to compare pulled to Ollama. You need to change the model names below to the ones on your computer (run 'ollama list' in your terminal to get the correct names).

'''

import os

import pandas as pd
from langchain_ollama import OllamaEmbeddings

# Import tools for computing the values from the 'tools' folder
from tools.calculate_cosine_similarity import cosine_similarity
from tools.euclidean_distance import calculate_euclidean_distance
from tools.l1_distance import calculate_l1_distance

# Check that the model names correspond to the ones installed on your device
for OLLAMA_MODEL in ["bge-m3:F16", "snowflake-arctic-embed-l-v2.0:F16", "embeddinggemma:F32", "multilingual-e5-large:F16", "nomic-embed-text-v2-moe:F32", "qwen3-embedding-8b:Q6_K"]:
    
    # Short model name for savefile
    MODEL_FILETITLE = OLLAMA_MODEL[0:5]
    
    # Model name for the results in the file; remove the quantization info from the name
    MODEL_NAME = ""
    if OLLAMA_MODEL == "qwen3-embedding-8b:Q6_K":
        MODEL_NAME = "qwen3-embedding-8b"
    else:
        MODEL_NAME = OLLAMA_MODEL[0:len(OLLAMA_MODEL)-4]

    # Default Ollama local server
    OLLAMA_URL = "http://127.0.0.1:11434" 

    # Directory for cases and path for result file for the model
    BASE_DIR = "cases"
    OUTPUT_PATH = "results\\results_" + MODEL_FILETITLE + ".csv"

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
                        cosine = cosine_similarity(embedding, original_text, mod_text)
                        euclidean = calculate_euclidean_distance(
                            embedding, original_text, mod_text
                        )
                        l1 = calculate_l1_distance(embedding, original_text, mod_text)

                        results.append(
                            {
                                "model": MODEL_NAME,
                                "case": case_dir,
                                "language": LANGUAGE,
                                "type": variation,
                                "file": file,
                                "cosine_similarity": cosine,
                                "euclidean_distance": euclidean,
                                "l1_distance": l1,
                            }
                        )
                        print(f"{case_dir} / {file} processed.")
                    except Exception as e:
                        print(f"Error processing {file}: {e}")

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_PATH, index=False, sep=";")
    print(f"\n All results saved to: {OUTPUT_PATH}")
