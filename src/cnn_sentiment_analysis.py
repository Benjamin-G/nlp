import os
import sys

import pandas as pd
from keras import Sequential
from keras.layers import Convolution1D, Dense, Dropout, Embedding, Flatten
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences, plot_model
from sklearn.model_selection import train_test_split

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parent_dir)

from src.utils import get_data_path


def run():
    # Load the data and map the tags to 1 and 0
    fields = ["text", "tag"]
    data = pd.read_csv(get_data_path("movie_review.csv"), usecols=fields)
    data["label"] = data["tag"].map({"pos": 1, "neg": 0}).astype("string")

    dtypes = {
        "text": "string",
        "tag": "string",
        "label": "string",
    }

    data = data.astype(dtypes)

    sample_data = data.sample(frac=0.25)

    max_words = 1000
    tokenizer = Tokenizer(num_words=max_words, split=" ")
    tokenizer.fit_on_texts(sample_data["text"].values)

    X = tokenizer.texts_to_sequences(sample_data["text"].values)
    X = pad_sequences(X)
    Y = pd.get_dummies(sample_data["label"]).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=36)

    print("Test data:")
    print(sample_data.info())
    print(round(sys.getsizeof(X_train) / (1024 * 1024 * 1024), 2), "GB")
    print(sys.getsizeof(y_train), "bytes")
    print(len(X_train))
    print(X.shape[1])
    # print(X_train)

    embedding_vector_length = 100

    model = Sequential()
    model.add(
        Embedding(
            max_words,
            embedding_vector_length,
            input_length=X.shape[1],
        ),
    )
    # The values (64,32,16) were chosen arbitrarily; good practice would be to estimate these values
    # (hyperparameters) on some held-out validation data.
    model.add(Convolution1D(64, 3, padding="same"))
    # Since the filter length (the kernel size) is 3, we start with the first three words (or rows in the matrix)
    model.add(Convolution1D(32, 3, padding="same"))
    # The second convolutional layer receives this 2D matrix as input. Since it is a 1D convolutional layer,
    # it applies its 32-fold convolution to the 65 64-dimensional entries.
    model.add(Convolution1D(16, 3, padding="same"))
    # Every layer specifies the dimensionality of the output space (64,32,16) and the size of every filter (3),
    # also known as the kernel size.
    model.add(Flatten())
    model.add(Dropout(0.2))

    # Dense layers connect all incoming neurons from a previous layer to 
    # neurons in the next layer; they are fully connected. 
    model.add(Dense(2, activation="sigmoid"))

    model.summary()

    # Export to png
    plot_model(model, to_file="../data/cnn_sentiment_analysis.png")

    # we “compile” it. This means we prepare the for training by specifying a loss function (a real-valued function
    # that computes the mismatch between predictions the model makes and the labels it should assign according to the
    # training data) plus a numerical optimizer algorithm carrying out the gradient descent process and an evaluation
    # metric, which is used to perform an intermediate evaluation of the model during training and which specifies
    # the loss of the loss function (like accuracy, or mean squared error).
    model.compile(
        loss="binary_crossentropy",
        # inary cross-entropy. Cross-entropy-based loss expresses how well a classification model performs
        # when producing probabilities (values between 0 and 1).
        optimizer="adam",
        metrics=["accuracy"],
    )

    # In the “fit” step, we can determine the size of this held-out portion of the training data (
    # validation_split=0.1: use 10% of the training data for testing), the number of training epochs (epochs=10), 
    # and the size of the batches of training data that are taken into account at every training step (batch_size=32).
    model.fit(
        X_train,
        y_train,
        # An epoch represents a full sweep through the training data. 
        epochs=10,
        batch_size=64,
    )

    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))


if __name__ == "__main__":
    run()
