import re
from os import listdir
from os.path import isfile, join

import numpy as np
from keras.utils import np_utils, pad_sequences
from keras_preprocessing.text import hashing_trick
from nltk import ngrams


def create_label_dict(path_training, path_test):
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
    d = d1.copy()
    d.update(d2)
    return d


def segment_document_words(filename, nb_words_per_segment):
    """
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
    with open(filename) as f:
        for line in f:
            tokens = line.rstrip("-\n").rstrip().split(" ")
            for token in tokens:
                if token != "":
                    words.append(token)
                    wordsDict[token] = 1

    f.close()
    segments = [words[i: i + nb_words_per_segment] for i in range(0, len(words), nb_words_per_segment)]
    return segments, len(wordsDict)


def vectorize_documents_BOW(path, label_dict, nb_words_per_segment):
    """
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
    with open(filename) as f:
        for line in f:
            ngram_list = ngrams(line.rstrip(), ngram_size)
            for ngram in ngram_list:
                joined = "_".join(ngram)
                words.append(joined)
                wordsDict[joined] = 1
    f.close()
    segments = [words[i:i + nb_words_per_segment] for i in range(0, len(words), nb_words_per_segment)]
    return segments, wordsDict


def vectorize_documents_ngrams(path, ngram_size, label_dict, nb_words_per_segment):
    """
    The vectorization procedure for word n-grams is then exactly the same as for separate words 
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


def run():
    training_dir = "../data/pan12-training"
    testing_dir = "../data/pan12-testing"

    print(training_dir, testing_dir)


if __name__ == "__main__":
    run()
