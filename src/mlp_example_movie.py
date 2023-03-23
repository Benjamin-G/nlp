import pandas as pd
from keras.layers.core import Activation, Dense
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils

from src.utils import get_data_path

if __name__ == "__main__":
    # Load the data and map the tags to 1 and 0
    fields = ["text", "tag"]
    data = pd.read_csv(get_data_path("movie_review.csv"), usecols=fields)
    data["label"] = data["tag"].map({"pos": 1, "neg": 0})
    data = data.sample(n=1000)
    docs = data["text"]

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(docs)

    # Convert the text to a matrix of token counts
    X_train = tokenizer.texts_to_matrix(docs, mode="binary")
    y_train = np_utils.to_categorical(data["label"])

    # Build the model
    input_dim = X_train.shape[1]
    nb_classes = y_train.shape[1]

    model = Sequential()
    model.add(Dense(128, input_dim=input_dim))

    # add 10 layers
    for _ in range(10):
        model.add(Activation("sigmoid"))
        model.add(Dense(128))

    model.add(Dense(nb_classes))
    model.add(Activation("softmax"))
    model.compile(loss="binary_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])

    print("Training...")
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1,
              shuffle=False, verbose=2)
