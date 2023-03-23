import pandas as pd
from keras.layers.core import Activation, Dense
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils

from src.utils import get_data_path

# log the device placement
# tf.debugging.set_log_device_placement(True)

if __name__ == "__main__":
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

    # Split the data into training and test sets
    # my laptop does not have much memory
    training_data = data.sample(frac=0.25)
    rest_part_25 = data.drop(training_data.index)

    # Preparing the training data
    docs = training_data["text"]

    # 
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(docs)

    # Convert the text to a matrix of token counts
    X_train = tokenizer.texts_to_matrix(docs, mode="binary")
    y_train = np_utils.to_categorical(training_data["label"])

    # Build the model
    input_dim = X_train.shape[1]
    nb_classes = y_train.shape[1]

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

    # add 10 layers ReLU
    for _ in range(10):
        model.add(Activation("relu"))
        model.add(Dense(128))
    # Epoch 10/10 29/29 - 0s - loss: 4.2112e-05 - accuracy: 1.0000 - val_loss: 5.0735 - val_accuracy: 0.6000 - 
    # 332ms/epoch - 11ms/step

    model.add(Dense(nb_classes))
    model.add(Activation("softmax"))
    model.compile(loss="binary_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])

    print("Training...")
    model.fit(
        X_train,
        y_train,
        epochs=8,
        # batch_size=256,
        validation_split=0.1,
        shuffle=False,
        verbose=2,
        use_multiprocessing=True,
    )
