from promptflow.core import tool
import numpy as np
import pickle

@tool
def coordinate_changes(filename1: str, filename2: str, min_max_first: str = "first", how_many: int = 1) -> float:
    """
    # GA Coordinate changes

    Loads two embedding vectors from given pickle files and computes the absolute values of the coordinate changes between the vectors. Considers the changes for the biggest, smallest or first listed changes, depending on the parameter `min_max_first`. Returns a list of given number of tuples where the first value is the absolute value of the coordinate change and the second value is the corresponding coordinate. The coordinates correspond to the position in the Python list (that is, the smallest coordinate is 0) instead of mathematical coordinates (which we can recover by simply adding 1 to the value).

    For example, compute_truncated_l1_and_pick_coordinates(model, fliename1, filename2, "min", 100) loads embedding vectors from pickle files `filename1` and `filename2`, then computes the absolute values of the coordinate changes between the embedding vectors, and returns a list of tuples containing the 100 smallest changes and the coordinates where those changes happen.

    :param: filename1 (str): The name of the pickle file for the first embedding vector.
    :param: filename2 (str): The name of the pickle file for the second embedding vector.
    :param: min_max_first (str): The accepted values are "min" (for the smallest changes), "max" (for the biggest changes), and "first" (for the first listed changes).
    :param: how_many (int): The number of entries in the output list.

    :return: The sum of the smallest or biggest or first changes between the embedding vectors of two texts.
    :rtype: list[tuple]
    """

    # check that the `min_max_all` parameter is correct
    if min_max_first != "min" and min_max_first != "max" and min_max_first != "first":
        raise ValueError('Incorrect value for the parameter min_max_first.')
    
    # get the embedding vectors and compute the coordinate changes
    vector1 = []
    with open(filename1, "rb") as fp1:   # Unpickling
        a = pickle.load(fp1)
        vector1 = np.array(a)

    vector2 = []
    with open(filename2, "rb") as fp2:   # Unpickling
        b = pickle.load(fp2)
        vector2 = np.array(b)

    distances = np.absolute(vector1 - vector2)
    coordinates = list(range(0,len(distances)))
    dis_coor = list(zip(distances, coordinates))

    # check that the `how_many` parameter is correct
    if how_many < 1 or how_many > len(dis_coor):
        raise ValueError('Incorrect value for the parameter how_many.')
        
    if min_max_first == "first":
        return dis_coor[0:how_many]
    
    if min_max_first == "min":
        sorte = sorted(dis_coor, key=lambda tup: tup[0])
        return sorte[0:how_many]
    
    if min_max_first == "max":
        sorte = sorted(dis_coor, key=lambda tup: tup[0], reverse=True)
        return sorte[0:how_many]