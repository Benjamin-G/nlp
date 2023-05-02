import codecs
import random
import re

import numpy as np
import pandas as pd
from keras import Input, Model
from keras.layers import Dense, Embedding, Reshape, dot
from keras.preprocessing.sequence import skipgrams
from sklearn.model_selection import train_test_split

from src.utils import create_vocabulary, list_vectors, save_embeddings


def save_embedding(outputFile, weights, vocabulary):
    rev = {v: k for k, v in vocabulary.items()}
    with codecs.open(outputFile, "w") as f:
        f.write(str(len(vocabulary)) + " " + str(weights.shape[1]) + "\n")
        for index in sorted(rev.keys()):
            word = rev[index]
            f.write(str(word) + " ")
            for i in range(len(weights[index])):
                f.write(str(weights[index][i]) + " ")
            f.write("\n")


def process_data(df, window_size):
    couples = []
    labels = []
    sentences = df["text"].tolist()
    vocab = {}
    vocab = create_vocabulary(vocab, sentences)
    vocab_size = len(vocab)
    for _, row in df.iterrows():
        words = []
        labels.append(row["label"])
        for w in row["text"].split(" "):
            w = re.sub("[.,:;'\"!?()]+", "", w.lower())
            if w != "":
                words.append(vocab[w])
        c, l = skipgrams(words, vocab_size, window_size=window_size)
        couples.extend(c)
        labels.extend(l)

    return vocab, couples, labels


def generator(target, context, labels, batch_size):
    """
    A special generator function loops over these samples during the training phase of the model, 
    picking out random samples and labels and feeding them as batches to the model.
    :param target: 
    :type target: 
    :param context: 
    :type context: 
    :param labels: 
    :type labels: 
    :param batch_size: 
    :type batch_size: 
    :return: 
    :rtype: 
    """
    batch_target = np.zeros((batch_size, 1))
    batch_context = np.zeros((batch_size, 1))
    batch_labels = np.zeros((batch_size, 1))

    while True:
        for i in range(batch_size):
            index = random.randint(0, len(target) - 1)
            batch_target[i] = target[index]
            batch_context[i] = context[index]
            batch_labels[i] = labels[index]

        yield [batch_target, batch_context], np.array(batch_labels)


EMBEDDING_PATH = "../data/test/w2v_embedding.txt"


def run():
    # dataset = load_dataset("yelp_polarity") from HuggingFace
    df = pd.read_parquet("../data/yelp_polarity.parquet.gzip")
    df = df.sample(frac=0.05)
    train, test = train_test_split(df, test_size=0.2)
    print(train.info())
    print(test.info())
    window_size = 3
    vector_dim = 100

    vocab, couples, labels = process_data(train, window_size)

    vocab_size = len(vocab)

    word_target, word_context = zip(*couples)

    print("Vocab size: ", vocab_size)

    input_target = Input((1,))
    input_context = Input((1,))

    # https://wikipedia2vec.github.io/wikipedia2vec/pretrained/

    embedding = Embedding(
        vocab_size,
        vector_dim,
        input_length=1,
    )

    # Our network is a simple combination of two Embedding layers 
    # (one for the source words and one for their context words) 
    # feeding into a dense (regular, fully connected) layer to which the output labels are fed 
    # (0 if the context word is not a valid context for the source word or 1 if it is). 
    target = embedding(input_target)
    target = Reshape((vector_dim, 1))(target)
    context = embedding(input_context)
    context = Reshape((vector_dim, 1))(context)

    dot_product = dot([target, context], axes=1, normalize=False)
    dot_product = Reshape((1,))(dot_product)
    output = Dense(1, activation="sigmoid")(dot_product)
    model = Model(inputs=[input_target, input_context], outputs=output)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])

    print(model.summary())

    # But to obtain truly meaningful and convincing results, 
    # the model should be trained on much larger amounts of data, for many more iterations.
    model.fit(
        generator(word_target, word_context, labels, 100),
        steps_per_epoch=100,
        epochs=100,
    )

    save_embeddings(EMBEDDING_PATH, embedding.get_weights()[0], vocab)


if __name__ == "__main__":
    run()
    list_vectors(EMBEDDING_PATH)
