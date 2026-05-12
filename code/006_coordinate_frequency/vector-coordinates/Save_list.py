from promptflow.core import tool
import pickle

@tool
def save_list(data: list, filename: str):
    """
    # GA Save List

    Saves the input list to a pickle file with the given name.

    :param: data (list): The list to save.
    :param: filename (str): Name of the file where the list is saved.
    """

    with open(filename, "wb") as fp:
        pickle.dump(data, fp)