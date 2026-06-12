# Clinical Note Comparison and Data Retrieval Via Embedding Vectors: Model Selection, Metrics, and Convergence

Code repository accompanying the manuscript for comparing clinical texts using embedding vectors and mathematical operations, with human review validation.
Tool kit referenced: https://github.com/HYGenAID/Tools.git
---

## Repository Structure

```
/
├── cases/                              # Synthetic discharge summaries and their perturbed versions
├── code/                               # Scripts for reproducing the results (see instructions below)
│   ├── 001_raw_data_for_open_mod/      # Raw data processing for open models
│   ├── 002_raw_data_for_closed_mo/     # Raw data processing for closed models
│   ├── 003_conflicting_changes/        # Conflicting changes analysis
│   ├── 004_convergence_analysis/       # Convergence analysis
│   ├── 005_pca/                        # PCA analysis
│   ├── 006_coordinate_frequency/       # Coordinate frequency analysis
│   └── 007_visualisations_and_tables/  # Code for visualisations and tables
├── genaid-promptflow/                  # Additional tool package for Prompt flow and requirement file
└── README.md
```

---

## Requirements for the scripts

The scripts in the code folder require the following programs, packages, and adjustments.

1) **Python** version 3.11.9 (most scripts work for later versions but there might be some challenges).

2) **Microsoft Visual Studio Code** (VS Code; version 1.112.0 was used for the manuscript) and its **Prompt flow** extension (version 1.18.0 was used for the manuscript).

3) Installing the **genaid-promptflow** package version 1.2.0 (see https://github.com/HYGenAID/Tools for more information about the package). Download the files `genaid_promptflow-1.2.0b1-py3-none-any.whl` and `requirements.txt` from the `genaid-promptflow/` folder in this repository. First file contains an additional tool package for Prompt flow, and the second file contains a list of the suitable versions for other packages. Install these packages in the download location by running the commands

```bash
pip install genaid_promptflow-1.2.0b1-py3-none-any.whl
```
and
```bash
python -m pip install -r "requirements.txt"
```

4) Installing Ollama (https://ollama.com/, version 0.13.2 was used in the manuscript) and pulling the open-source embedding models from Hugging Face (https://huggingface.co/). For example, the manuscript uses the quantization Q6_K of Qwen3-Embedding-8B. The model is available on Hugging Face https://huggingface.co/Qwen/Qwen3-Embedding-8B-GGUF and the corresponding quantization is listed on the right hand side under GGUF. Clicking the quantization name opens information about the quantization. Click the `Use this model` pull-down menu and choose Ollama. This shows you the command for running the model:
```bash
ollama run hf.co/Qwen/Qwen3-Embedding-8B-GGUF:Q6_K
```
Using this command pulls the model on your device (and then tries to use it, but this results in error; use `pull` instead of `run` to avoid this). After pulling the model, run the command
```bash
ollama list
```
to verify that it is on your device. The names of the models in the list are the ones you need to use in the scripts. In the manuscript, we used the following open-source models and quantizations on Ollama:
- bge-m3, F16
- EmbeddingGemma, F32
- multilingual-e5-large, F16
- Nomic Embed v2, F32
- Qwen3-Embedding-8B, Q6_K
- Snowflake Arctic Embed L v2.0, F16

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for exact model digests, engine versions, and Azure deployment access dates.

1) For OpenAI's closed-source models, you have to set up an **Azure OpenAI** connection in Prompt flow. Open Prompt flow on VS Code, scroll all the way down in the left hand side menus to "Connections" and set up the connection. You need OpenAI API key for this. Follow the instructions and remember the name of your connection.

---

## Running the scripts

All the scripts used to generate the results in the manuscript are available in the `code/` folder. The scripts are organized into numbered subfolders and they should be run in order to generate the result files in correct order. Similarly, the scripts in each folder are also numbered and they should be run in order. Before running each script, open it and type in the directory or file paths that the scripts require. In addition, check that the names of the embedding models in certain scripts correspond to the ones installed on your Ollama and remove any models that you are not using (e.g., closed-source models).

---

## Data Availability

The synthetic discharge summaries and their perturbed versions can be found in the `cases/` folder in this repository. Source data for the human review is not included in this repository due to patient privacy constraints. 

---

## Citations

*Harris CR, Millman KJ, van der Walt SJ, et al. Array programming with NumPy. Nature 585*, 357–362 (2020). [https://doi.org/10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2)

*Hunter JD. Matplotlib: A 2D Graphics Environment. Computing in Science & Engineering 9*, 90–95 (2007). [https://doi.org/10.1109/MCSE.2007.55](https://doi.org/10.1109/MCSE.2007.55)

Jinja [software on the Internet]. Available from: [https://pypi.org/project/Jinja2/] (https://pypi.org/project/Jinja2/)

*LangChain Developers. LangChain. PyPI* (2025). Available from: [https://pypi.org/project/langchain/](https://pypi.org/project/langchain/)


*Microsoft Corporation. Microsoft Azure SDK for Python. PyPI* (2025). Available from: [https://pypi.org/project/azure/](https://pypi.org/project/azure/)

*Pallets team. Jinja. PyPI* (2025). Available from: [https://pypi.org/project/Jinja2/](https://pypi.org/project/Jinja2/)

*Pedregosa F, Varoquaux G, Gramfort A, et al. Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research 12*, 2825–2830 (2011). [http://www.jmlr.org/papers/v12/pedregosa11a.html](http://www.jmlr.org/papers/v12/pedregosa11a.html)

*Virtanen P, Gommers R, Oliphant TE, et al. SciPy 1.0: fundamental algorithms for scientific computing in Python. Nature Methods 17*, 261–272 (2020). [https://doi.org/10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2)

*Waskom ML. seaborn: statistical data visualization. Journal of Open Source Software 6*, 3021 (2021). [https://doi.org/10.21105/joss.03021](https://doi.org/10.21105/joss.03021)

*Wolf T, Debut L, Sanh L, et al. Transformers: State-of-the-Art Natural Language Processing. Proceedings of EMNLP 2020: System Demonstrations. Association for Computational Linguistics*, 38–45 (2020). [https://doi.org/10.18653/v1/2020.emnlp-demos.6](https://doi.org/10.18653/v1/2020.emnlp-demos.6)

---

## Citation

If you use this repository, please cite:

```bibtex
@article{
  title    = {Clinical Note Comparison and Data Retrieval Via Embedding Vectors: Model Selection, Metrics, and Convergence},
  author   = {Alexandra Dahlberg, Olli Tapiola, Rami Luisto, Tuukka Puranen, Enni Sanmark and Ville Vartiainen},
  year     = {2026},
  preprint = {https://doi.org/10.64898/2026.05.12.26352832}
}
```



