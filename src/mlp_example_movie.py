import argparse
import sys

import pandas as pd
from keras import models
from keras.layers.core import Activation, Dense
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils

from utils import SPACER, get_data_path, get_models_path

# log the device placement
# tf.debugging.set_log_device_placement(True)

# parsers if needed
parser = argparse.ArgumentParser()
parser.add_argument("-T", "--test", action=argparse.BooleanOptionalAction)
args = parser.parse_args()


#
# print(args.new)


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
    print("Loaded data:")
    print(data.info())
    print(SPACER)

    # Split the data into training and test sets
    # my laptop does not have much memory
    sample_data = data.sample(frac=0.25)

    # Preparing the training data
    sample_docs = sample_data["text"]

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sample_docs)

    # Convert the text to a matrix of token counts
    x = tokenizer.texts_to_matrix(sample_docs, mode="binary")
    y = np_utils.to_categorical(sample_data["label"])

    print("Test data:" if args.test else "Sample data:")
    print(sample_data.info())
    print(round(sys.getsizeof(x) / (1024 * 1024 * 1024), 2), "GB")
    print(sys.getsizeof(y), "bytes")
    print(len(x))
    print(len(x[0]))
    print(SPACER)

    # Run the test
    # Still not working
    if args.test:
        model = models.load_model(get_models_path("mlp_example_movie.h5"))
        model.summary()
        print(SPACER + "Testing..." + SPACER)
        prediction = model.predict(
            x,
        )
        print("prediction shape:", prediction.shape)
        print(prediction)
        # for i in range(10):
        #     prediction = model.predict(np.array([x[i]]))
        #     predicted_label = np.argmax(prediction[0])
        #     print("Actual label:" + x.iloc[i])
        #     print("Predicted label: " + predicted_label)
        #     print(prediction)
        #     print(SPACER)

        return

    # Build the model
    input_dim = x.shape[1]
    nb_classes = y.shape[1]

    # a Sequential model, which defines a container for a stacked set of layers, and facilities for defining layers.
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim))

    # add 10 layers sigmoid 
    # We observe that the network doesn`t seem to learn at all: its validation accuracy (the 
    # accuracy it attains on a held-out test portion of its training data during training) does not increase.
    # for _ in range(10):
    #     model.add(Activation("sigmoid"))
    #     model.add(Dense(128))
    # Epoch 10/10
    # 29/29 - 0s - loss: 0.6973 - accuracy: 0.4933 - val_loss: 0.6969 - val_accuracy: 0.4600 - 
    # 345ms/epoch - 12ms/step

    # networks with more than two layers between their input and output layer may be deemed deep.
    # add 10 layers ReLU
    for _ in range(10):
        model.add(Activation("relu"))
        model.add(Dense(128))
    # Epoch 10/10 29/29 - 0s - loss: 4.2112e-05 - accuracy: 1.0000 - val_loss: 5.0735 - val_accuracy: 0.6000 - 
    # 332ms/epoch - 11ms/st

    model.add(Dense(nb_classes))
    model.add(Activation("softmax"))
    model.compile(loss="binary_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])

    model.summary()

    print(SPACER + "Training..." + SPACER)
    history = model.fit(
        x,
        y,
        epochs=8,
        batch_size=256,
        validation_split=0.1,
        shuffle=False,
        verbose=2,
        # use_multiprocessing=True,
    )

    hist_df = pd.DataFrame(history.history)
    print(SPACER)
    print(hist_df)

    print(SPACER + "Saving Model..." + SPACER)
    model.save(get_models_path("mlp_example_movie.h5"))


if __name__ == "__main__":
    run()
