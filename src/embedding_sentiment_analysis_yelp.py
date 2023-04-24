import codecs
import os
import re

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from keras.layers import Dense, Embedding, Flatten
from keras.models import Sequential
from keras.utils import pad_sequences
from sklearn.model_selection import train_test_split


def save_embedding(outputFile, weights, vocabulary):
    rev = {v: k for k, v in vocabulary.items()}
    # rev = vocabulary
    with codecs.open(outputFile, "w") as f:
        f.write(str(len(vocabulary)) + " " + str(weights.shape[1]) + "\n")
        for index in sorted(rev.keys()):
            word = rev[index]
            f.write(str(word) + " ")
            for i in range(len(weights[index])):
                f.write(str(weights[index][i]) + " ")
            f.write("\n")


def create_vocabulary(vocabulary, sentences):
    vocabulary["<unk>"] = 0
    for sentence in sentences:
        for word in sentence.strip().split():
            word = re.sub("[.,:;'\"!?()]+", "", word.lower())
            if word not in vocabulary:
                vocabulary[word] = len(vocabulary)

    return vocabulary


def process_training_data(df, max_len):
    data = []
    vocab = {}
    labels = []
    sentences = df["text"].tolist()
    vocab = create_vocabulary(vocab, sentences)
    for _, row in df.iterrows():
        words = []
        labels.append(row["label"])
        for w in row["text"].split(" "):
            w = re.sub("[.,:;'\"!?()]+", "", w.lower())
            if w != "":
                words.append(vocab[w])
        data.append(words)

    data = pad_sequences(data, maxlen=max_len, padding="post")

    return data, np.array(labels), vocab


def process_test_data(df, vocab, max_len):
    data = []
    labels = []
    sentences = df["text"].tolist()
    vocab = create_vocabulary(vocab, sentences)
    for _, row in df.iterrows():
        words = []
        labels.append(row["label"])
        for w in row["text"].split(" "):
            w = re.sub("[.,:;'\"!?()]+", "", w.lower())
            if w != "":
                if w in vocab:
                    words.append(vocab[w])
                else:
                    words.append(vocab["<unk>"])
        data.append(words)
    data = pad_sequences(data, maxlen=max_len, padding="post")

    return data, np.array(labels)


EMBEDDING_PATH = "../data/test/yelp_embedding_labeled.txt"


def run():
    df = pd.read_parquet("../data/yelp_polarity.parquet.gzip")
    df = df.sample(frac=0.15)
    train, test = train_test_split(df, test_size=0.2)
    print(train.info())
    print(test.info())
    max_len = 100
    data, labels, vocab = process_training_data(train, max_len)
    test_data, test_labels = process_test_data(test, vocab, max_len)

    model = Sequential()
    embedding = Embedding(len(vocab), 100, input_length=max_len)
    model.add(embedding)
    model.add(Flatten())
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])
    model.fit(data, labels, epochs=100, verbose=1)

    loss, accuracy = model.evaluate(test_data, test_labels, verbose=0)
    print(accuracy)

    save_embedding(EMBEDDING_PATH, embedding.get_weights()[0], vocab)


def analysis():
    w2v = KeyedVectors.load_word2vec_format(
        os.path.join(EMBEDDING_PATH), binary=False, unicode_errors="ignore")

    for w in sorted(w2v.wv.vocab):
        print(w, w2v.most_similar(w, topn=3))


if __name__ == "__main__":
    run()
    analysis()
