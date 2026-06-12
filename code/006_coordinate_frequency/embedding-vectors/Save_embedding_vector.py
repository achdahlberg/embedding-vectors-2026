from promptflow.core import tool
from langchain_core.embeddings.embeddings import Embeddings
import pickle

@tool
def save_embedding_vector(embeddings: Embeddings, text: str, filename: str):
    """
    # GA Save Embedding Vector

    Saves the embedding vector of the given text to a pickle file with the given name. The embedding model is integrated to the `Embeddings` input.

    :param: embeddings (Embeddings): The `Embeddings` model for converting the texts into vector form.
    :param: text (str): The text to convert into vector form.
    :param: filename (str): Name of the file where the embedding vector is saved.
    """

    vector = list(embeddings.embed_query(text))

    with open(filename, "wb") as fp:
        pickle.dump(vector, fp)