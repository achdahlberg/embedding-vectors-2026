import numpy as np
from langchain_core.embeddings.embeddings import Embeddings
from numpy.linalg import norm
from promptflow.core import tool


@tool
def calculate_euclidean_distance(embeddings: Embeddings, text1: str, text2: str) -> float:
    """
    # GA Euclidean Distance

    Gives the Euclidean distance between the embedding vectors of two texts.

    :param: embeddings (Embeddings): The `Embeddings` model for converting the texts into vector form.
    :param: text1 (str): The first text to convert.
    :param: text2 (str): The second text to convert.

    :return: The Euclidean distance between the embedding vectors of the two input texts.
    :rtype: float
    """
    vector1 = embeddings.embed_query(text1)
    vector2 = embeddings.embed_query(text2)

    a = np.array(vector1)
    b = np.array(vector2)
    
    return norm(a-b)