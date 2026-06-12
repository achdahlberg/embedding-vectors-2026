import numpy as np
from promptflow.core import tool
from numpy.linalg import norm
from langchain_core.embeddings.embeddings import Embeddings

@tool
def cosine_similarity(embeddings: Embeddings, text1: str, text2: str) -> float:
    """
    # GA Cosine Similarity

    Gives the cosine similarity between the embedding vectors of two texts. The output value is between -1 and 1, where
     - -1 = vectors are pointing in the opposite directions,
     - 0 = vectors are orthogonal,
     - 1 = vectors are pointing in the same direction.

    :param: embeddings (Embeddings): The `Embeddings` model for converting the texts into vector form.
    :param: text1 (str): The first text to convert.
    :param: text2 (str): The second text to convert.

    :return: The cosine similarity value between the embedding vectors of the two input texts.
    :rtype: float
    """
    vector1 = embeddings.embed_query(text1)
    vector2 = embeddings.embed_query(text2)

    a = np.array(vector1)
    b = np.array(vector2)
    
    return np.dot(a,b)/(norm(a)*norm(b))