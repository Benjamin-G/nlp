import re
from os import listdir
from os.path import isfile, join

import numpy as np
from keras import Sequential
from keras.layers import Dense, Dropout, Embedding, Flatten
from keras.utils import np_utils, pad_sequences
from keras_preprocessing.text import hashing_trick
from nltk import ngrams
from sklearn.model_selection import train_test_split


def create_label_dict_one_file(path):
    files = [join(path, filename) for filename in listdir(path) if isfile(join(path, filename))]
    labelDict = {}

    for file in files:
        match = re.match("^.*\\/?12[A-Z][a-z]+([A-Z]+).+", file)
        if match:
            label = ord(match.group(1)) - 65
        else:
            print("Skipping filename:%s" % (file))
            continue
        if label not in labelDict:
            labelDict[label] = len(labelDict)

    return labelDict


def create_label_dict(path_training, path_test):
    """
    Creating a table dictionary
    :param path_training: 
    :type path_training: 
    :param path_test: 
    :type path_test: 
    :return: 
    :rtype: 
    """
    filesTraining = [join(path_training, filename) for filename in listdir(path_training) if
                     isfile(join(path_training, filename))]

    filesTest = [join(path_test, filename) for filename in listdir(path_test)
                 if isfile(join(path_test, filename))]
    files = filesTraining + filesTest

    labelDict = {}

    for file in files:
        match = re.match("^.*\\/?12[A-Z][a-z]+([A-Z]+).+", file)
        if match:
            # We convert every author identifier into a number by subtracting 65 (the ASCII value of A) from the 
            # ASCII value of the identifier. This creates a 0 for A, a 1 for B, and so on. Since we do not know if 
            # labels are missing in our data, we store these values in the dictionary, which maps every subtraction 
            # result to a unique index.
            label = ord(match.group(1)) - 65
        else:
            print("Skipping filename:%s" % file)
            continue
        if label not in labelDict:
            labelDict[label] = len(labelDict)

    return labelDict


def merge_dictionaries(d1, d2):
    """
    Merging dictionaries
    :param d1: 
    :type d1: 
    :param d2: 
    :type d2: 
    :return: 
    :rtype: 
    """
    d = d1.copy()
    d.update(d2)
    return d


def segment_document_words(filename, nb_words_per_segment):
    """
    Segmenting documents based on words
    segment size of nb_words_per_segment words produces a list of word-based segments
    :param filename: 
    :type filename: 
    :param nb_words_per_segment: 
    :type nb_words_per_segment: 
    :return: 
    :rtype: 
    """
    wordsDict = {}
    words = []
    with open(filename, encoding="cp437") as f:
        for line in f:
            tokens = line.rstrip("-\n").rstrip().split(" ")
            for token in tokens:
                if token != "":
                    words.append(token)
                    wordsDict[token] = 1

    segments = [words[i: i + nb_words_per_segment] for i in range(0, len(words), nb_words_per_segment)]

    return segments, wordsDict


def vectorize_documents_BOW(path, label_dict, nb_words_per_segment):
    """
    Vectorizing documents using bag-of-words
    This procedure generates word-based vectors for segmented documents.
    :param path: 
    :type path: 
    :param label_dict: 
    :type label_dict: 
    :param nb_words_per_segment: 
    :type nb_words_per_segment: 
    :return: 
    :rtype: 
    """
    files = [filename for filename in listdir(path) if isfile(join(path, filename))]
    segments = []
    labels = []
    globalDict = {}

    for file in files:
        match = re.match("^.*12[A-Z][a-z]+([A-Z]+).+", file)
        if match:
            label = ord(match.group(1)) - 65
        else:
            print("Skipping filename:%s" % file)
            continue

        segmented_document, wordDict = segment_document_words(join(path, file), nb_words_per_segment)

        globalDict = merge_dictionaries(globalDict, wordDict)

        segments.extend(segmented_document)

        for _segment in segmented_document:
            labels.append(label)

    vocab_len = len(globalDict)

    labels = [label_dict[x] for x in labels]
    nb_classes = len(label_dict)

    X = []

    for segment in segments:
        segment = " ".join(segment)
        X.append(
            pad_sequences(
                [
                    hashing_trick(
                        segment,
                        round(vocab_len * 1.3),
                    ),
                ],
                nb_words_per_segment,
            )[0],
        )

    y = np_utils.to_categorical(labels, nb_classes)

    return np.array(X), y, vocab_len


def segment_document_ngrams(filename, nb_words_per_segment, ngram_size):
    """
    Segmenting documents using word n-grams
    chop up sentences into word n-grams of a specified size
    :param filename:
    :type filename:
    :param nb_words_per_segment:
    :type nb_words_per_segment:
    :param ngram_size:
    :type ngram_size:
    :return:
    :rtype:
    """
    wordsDict = {}
    words = []
    with open(filename, encoding="cp437") as f:
        for line in f:
            ngram_list = ngrams(line.rstrip("-\n").rstrip(), ngram_size)
            for ngram in ngram_list:
                joined = "_".join(ngram)
                words.append(joined)
                wordsDict[joined] = 1

    segments = [words[i:i + nb_words_per_segment] for i in range(0, len(words), nb_words_per_segment)]

    return segments, wordsDict


def vectorize_documents_ngrams(path, ngram_size, label_dict, nb_words_per_segment):
    """
    Vectorizing documents using character n-grams
    
    The vectorization procedure for word n-grams is then exactly the same as for separate words
    
    If we preprocess our data explicitly by generating word n-grams ourselves, a CNN can detect higher-order n-grams 
    from these n-grams: n-grams of n-grams. 
    :param path: 
    :type path: 
    :param ngram_size: 
    :type ngram_size: 
    :param label_dict: 
    :type label_dict: 
    :param nb_words_per_segment: 
    :type nb_words_per_segment: 
    :return: 
    :rtype: 
    """
    files = [filename for filename in listdir(path) if isfile(join(path, filename))]
    segments = []
    labels = []
    globalDict = {}

    for file in files:
        match = re.match("^.*12[A-Z][a-z]+([A-Z]+).+", file)
        if match:
            label = ord(match.group(1)) - 65
        else:
            print("Skipping filename:%s" % file)
            continue

        segmented_document, wordDict = segment_document_ngrams(join(path, file), nb_words_per_segment, ngram_size)

        globalDict = merge_dictionaries(globalDict, wordDict)

        segments.extend(segmented_document)

        for _segment in segmented_document:
            labels.append(label)

    vocab_len = len(globalDict)

    labels = [label_dict[x] for x in labels]
    nb_classes = len(label_dict)

    X = []

    for segment in segments:
        segment = " ".join(segment)
        X.append(
            pad_sequences(
                [
                    hashing_trick(
                        segment,
                        round(vocab_len * 1.5)),
                ],
                nb_words_per_segment,
            )[0],
        )

    y = np_utils.to_categorical(labels, nb_classes)

    return np.array(X), y, int(vocab_len * 1.5) + 1


def segment_document_char_ngrams(filename, nb_words_per_segment, ngram_size):
    """
    Segmenting documents based on character n-grams
    
    a lot of research on authorship analysis (like Stamatatos 2009) has found that subword information
    like character n-grams also bears authorship-revealing information.
    :param filename:
    :type filename:
    :param nb_words_per_segment:
    :type nb_words_per_segment:
    :param ngram_size:
    :type ngram_size:
    :return:
    :rtype:
    """
    wordsDict = {}
    words = []
    with open(filename, encoding="cp437") as f:
        for line in f:
            line = line.rstrip("-\n").rstrip().replace(" ", "#")
            char_ngrams_list = ngrams(list(line), ngram_size)
            for char_ngram in char_ngrams_list:
                joined = "".join(char_ngram)
                words.append(joined)
                wordsDict[joined] = 1

    segments = [words[i:i + nb_words_per_segment] for i in range(0, len(words), nb_words_per_segment)]

    return segments, wordsDict


def vectorize_documents_char_ngrams(path, ngram_size, label_dict, nb_words_per_segment):
    """
    Vectorizing documents based on character n-grams
    :param path: 
    :type path: 
    :param ngram_size: 
    :type ngram_size: 
    :param label_dict: 
    :type label_dict: 
    :param nb_words_per_segment: 
    :type nb_words_per_segment: 
    :return: 
    :rtype: 
    """
    files = [filename for filename in listdir(path) if isfile(join(path, filename))]
    segments = []
    labels = []
    globalDict = {}

    for file in files:
        match = re.match("^.*12[A-Z][a-z]+([A-Z]+).+", file)
        if match:
            label = ord(match.group(1)) - 65
        else:
            print("Skipping filename:%s" % file)
            continue

        segmented_document, wordDict = segment_document_char_ngrams(join(path, file), nb_words_per_segment, ngram_size)

        globalDict = merge_dictionaries(globalDict, wordDict)

        segments.extend(segmented_document)
        for _segment in segmented_document:
            labels.append(label)

    vocab_len = len(globalDict)

    labels = [label_dict[x] for x in labels]
    nb_classes = len(label_dict)

    X = []

    for segment in segments:
        segment = " ".join(segment)
        X.append(
            pad_sequences(
                [hashing_trick(segment, round(vocab_len * 1.5))],
                nb_words_per_segment,
            )[0])

    y = np_utils.to_categorical(labels, nb_classes)

    return np.array(X), y, int(vocab_len * 1.5) + 1


def run():
    """
    The two models are evaluated for our 
    three types of data representation—word 
    unigrams, word n-grams, and character n-grams—on the PAN data.
    :return: 
    :rtype: 
    """
    train = "../data/pan12-training"
    # test = "../data/pan12-testing"

    labelDict = create_label_dict_one_file(train)

    input_dim = 500

    # Running this MLP on the single-word-based representation of the PAN data results in around 65% accuracy
    # X, y, vocab_len = vectorize_documents_BOW(train, labelDict, input_dim)

    # Performance drops significantly, to scores in the realm of 55% (54.17% in our run)
    # ngram_size above 2 starts having memory issues
    # Total params: 3,879,014 with 2... Total params: 1,261,872,864 with 10 !!!!
    X, y, vocab_len = vectorize_documents_char_ngrams(train, 2, labelDict, input_dim)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # nb_classes dimensionality (the number of authors in our dataset).
    nb_classes = len(labelDict)

    model = Sequential()
    # Embedding to represent the word indices of our 500-word blocks as 300-dimensional vectors.
    model.add(Embedding(vocab_len, 300, input_length=input_dim))
    model.add(Dense(300, activation="relu"))
    # Dropout layers randomly deactivate neurons in their input in order to avoid overfitting.
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(nb_classes, activation="sigmoid"))

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["acc"],
    )

    print(model.summary())

    nb_epochs = 10

    model.fit(
        X_train,
        y_train,
        epochs=nb_epochs,
        shuffle=True,
        batch_size=64,
        validation_split=0.3,
        verbose=2,
    )

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    print("Accuracy: %f" % (accuracy * 100))


if __name__ == "__main__":
    run()
