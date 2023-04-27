import os

import numpy as np
from gensim.models import KeyedVectors
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE


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


def tsne_plot(model, max_words=100):
    labels = []
    tokens = []

    n = 0
    for word in model:
        if n < max_words:
            tokens.append(model[word])
            labels.append(word)
            n += 1

    tsne_model = TSNE(
        perplexity=40,
        n_components=2,
        init="pca",
        n_iter=10000,
        random_state=23,
    )

    new_values = tsne_model.fit_transform(np.array(tokens))

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(8, 8))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords="offset points",
                     ha="right",
                     va="bottom")
    plt.show()
