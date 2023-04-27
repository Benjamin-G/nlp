import codecs
import re
import zipfile

import numpy as np
import pandas as pd
from keras.layers import Dense, Embedding, Flatten
from keras.models import Sequential
from keras.utils import pad_sequences
from sklearn.model_selection import train_test_split

from src.utils import get_data_path


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


def load_embedding_zipped(f, vocab, embedding_dimension):
    embedding_index = {}
    with zipfile.ZipFile(get_data_path(f)) as z:
        with z.open("glove.6B.100d.txt") as f:
            n = 0
            for line in f:
                if n:
                    values = line.split()
                    word = values[0]
                    if word in vocab:
                        coefs = np.asarray(values[1:], dtype='float32')
                        embedding_index[word] = coefs
                n += 1
    z.close()

    # Prepare embedding matrix
    embedding_matrix = np.zeros((len(vocab) + 1, embedding_dimension))
    hits = 0
    misses = 0
    for word, i in vocab.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            # This includes the representation for "padding" and "OOV"
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            misses += 1

    print("Converted %d words (%d misses)" % (hits, misses))
    return embedding_matrix


EMBEDDING_PATH = "../data/test/yelp_embedding_labeled.txt"


def run():
    # dataset = load_dataset("yelp_polarity") from HuggingFace
    df = pd.read_parquet("../data/yelp_polarity.parquet.gzip")
    df = df.sample(frac=0.15)
    train, test = train_test_split(df, test_size=0.2)
    print(train.info())
    print(test.info())
    max_len = 100
    data, labels, vocab = process_training_data(train, max_len)
    test_data, test_labels = process_test_data(test, vocab, max_len)

    vocab_size = len(vocab)
    print(vocab)
    print("Vocab size: ", vocab_size)

    # Build model
    model = Sequential()

    # Download GloVe from
    # http://nlp.stanford.edu/data/glove.6B.zip
    embedding_dimension = 100
    """
    pretrained external embeddings
    """

    embedding_matrix = load_embedding_zipped("glove.6B.zip", vocab, embedding_dimension)
    embedding = Embedding(len(vocab) + 1,
                          embedding_dimension,
                          weights=[embedding_matrix],
                          input_length=max_len,
                          trainable=False)

    """
    task-specific embeddings
    """
    embedding = Embedding(len(vocab), embedding_dimension, input_length=max_len)
    model.add(embedding)

    model.add(Flatten())
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])
    model.fit(data, labels, epochs=20, verbose=1)

    loss, accuracy = model.evaluate(test_data, test_labels, verbose=0)
    print(accuracy)

    save_embedding(EMBEDDING_PATH, embedding.get_weights()[0], vocab)


if __name__ == "__main__":
    run()
    # list_vectors(EMBEDDING_PATH)
