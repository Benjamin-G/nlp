import os

from gensim.models import KeyedVectors


def get_data_path(filename: str) -> str:
    """
    Get the path of the file in the data folder
    :param filename: name of the file
    :type filename: str
    :return: path of the file
    :rtype: str
    """
    return os.path.join(os.path.dirname(__file__), "..", "data", filename)


def get_models_path(filename: str) -> str:
    """
    Get the path of the file in the models folder
    :param filename: name of the file
    :type filename: str
    :return: path of the file
    :rtype: str
    """
    return os.path.join(os.path.dirname(__file__), "..", "models", filename)


SPACER = f'\n{"-" * 40}\n'


def list_vectors(EMBEDDING_PATH):
    """
    Our program lists the three nearest neighbors for every word based on its derived vector representation
    :return:
    :rtype:
    """
    w2v = KeyedVectors.load_word2vec_format(
        os.path.join(EMBEDDING_PATH), binary=False, unicode_errors="ignore")

    for w in sorted(w2v.key_to_index):
        print(w, w2v.most_similar(w, topn=3))
