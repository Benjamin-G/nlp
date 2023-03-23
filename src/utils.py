import os


def get_data_path(filename: str) -> str:
    """
    Get the path of the file in the data folder
    :param filename: name of the file
    :type filename: str
    :return: path of the file
    :rtype: str
    """
    return os.path.join(os.path.dirname(__file__), "..", "data", filename)
