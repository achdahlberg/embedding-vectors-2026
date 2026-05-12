from promptflow.core import tool
from langchain_core.embeddings.embeddings import Embeddings
from numpy.linalg import norm
import numpy as np

@tool
def compute_truncated_l2_distance(embeddings: Embeddings, text1: str, text2: str, min_max_first: str = "first", how_many: int = 1) -> float:
    """
    # GA Truncated L2 Distance

    Computes the truncated L2 distance between the embedding vectors of two texts, that is, the square root of the sum of squares of absolute values of a given number of coordinate changes between the vectors. Computes the value for the biggest, smallest or first listed changes, depending on the parameter `min_max_first`.

    For example, compute_average_truncated_l2_distance(model, text1, text2, "min", 100) computes the square root of the sum of squares of absolute values for the smallest 100 changes between the embedding vectors.

    :param: embeddings (Embeddings): The `Embeddings` model for converting the texts into vector form.
    :param: text1 (str): The first text to convert.
    :param: text2 (str): The second text to convert.
    :param: min_max_first (str): The accepted values are "min" (for the smallest changes), "max" (for the biggest changes), and "first" (for the first listed changes).
    :param: how_many (int): The number of changes used to compute the average.

    :return: The square root of the sum of squares of absolute values for the smallest or biggest or first changes between the embedding vectors of two texts.
    :rtype: float
    """

    # check that the `min_max_all` parameter is correct
    if min_max_first != "min" and min_max_first != "max" and min_max_first != "first":
        raise ValueError('Incorrect value for the parameter min_max_first.')
    
    # get the embedding vectors and compute the coordinate changes
    vector1 = np.array(embeddings.embed_query(text1))
    vector2 = np.array(embeddings.embed_query(text2))
    distances = np.absolute(vector1 - vector2)

    # check that the `how_many` parameter is correct
    if how_many < 1 or how_many > len(distances):
        raise ValueError('Incorrect value for the parameter how_many.')
        
    if min_max_first == "first":
        return norm(distances[0:how_many])
    
    if min_max_first == "min":
        distances.sort()
        return norm(distances[0:how_many])
    
    if min_max_first == "max":
        distances[::-1].sort()
        return norm(distances[0:how_many])