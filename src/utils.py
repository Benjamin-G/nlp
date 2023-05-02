import codecs
import os
import re

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


def list_vectors(embedding_path):
    """
    Our program lists the three nearest neighbors for every word based on its derived vector representation
    :return:
    :rtype:
    """
    w2v = KeyedVectors.load_word2vec_format(
        os.path.join(embedding_path), binary=False, unicode_errors="ignore")

    res = ""
    for w in sorted(w2v.key_to_index):
        res += f"{w}: {w2v.most_similar(w, topn=3)}\n"
        # print(w, w2v.most_similar(w, topn=3))

    with open("../data/test/list_vectors.txt", "w") as text_file:
        print("---- saving list_vectors.txt")
        text_file.write(res)


def tsne_plot(model, max_words=100):
    print(model)
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


def create_vocabulary(vocabulary, sentences):
    vocabulary["<unk>"] = 0
    for sentence in sentences:
        for word in sentence.strip().split():
            word = re.sub("[.,:;'\"!?()]+", "", word.lower())
            if word not in vocabulary:
                vocabulary[word] = len(vocabulary)

    return vocabulary


def save_embeddings(outputFile, weights, vocabulary):
    rev = {v: k for k, v in vocabulary.items()}
    with codecs.open(outputFile, "w") as f:
        f.write(str(len(vocabulary)) + " " + str(weights.shape[1]) + "\n")
        for index in sorted(rev.keys()):
            word = rev[index]
            f.write(str(word) + " ")
            for i in range(len(weights[index])):
                f.write(str(weights[index][i]) + " ")
            f.write("\n")
