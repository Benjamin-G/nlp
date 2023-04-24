import numpy as np
from keras.layers import Dense, SimpleRNN, TimeDistributed
from keras.models import Sequential
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def run():
    data = ["character"]
    enc = LabelEncoder()
    alphabet = np.array(list({c for w in data for c in w}))  #
    enc.fit(alphabet)
    int_enc = enc.fit_transform(alphabet)
    onehot_encoder = OneHotEncoder(sparse_output=False)
    int_enc = int_enc.reshape(len(int_enc), 1)
    onehot_encoder.fit_transform(int_enc)

    X_train = []
    y_train = []

    for w in data:
        for i in range(len(w) - 1):
            X_train.extend(onehot_encoder.transform([enc.transform([w[i]])]))
            y_train.extend(onehot_encoder.transform([enc.transform([w[i + 1]])]))

    X_test = []
    y_test = []

    test_data = ["character"]

    for w in test_data:
        for i in range(len(w) - 1):
            X_test.extend(onehot_encoder.transform([enc.transform([w[i]])]))
            y_test.extend(onehot_encoder.transform([enc.transform([w[i + 1]])]))

    sample_size = 256
    sample_len = len(X_train)

    X_train = np.array([X_train * sample_size]).reshape(sample_size, sample_len, len(alphabet))
    y_train = np.array([y_train * sample_size]).reshape(sample_size, sample_len, len(alphabet))
    test_len = len(X_test)
    X_test = np.array([X_test]).reshape(1, test_len, len(alphabet))
    y_test = np.array([y_test]).reshape(1, test_len, len(alphabet))

    model = Sequential()
    model.add(
        SimpleRNN(
            units=100,
            input_dim=len(alphabet),
            # output_dim=100,
            return_sequences=True,
        ),
    )
    model.add(
        TimeDistributed(
            Dense(
                len(alphabet),
                # output_dim=len(alphabet),
                activation="sigmoid",
            ),
        ),
    )
    model.compile(
        loss="binary_crossentropy",
        metrics=["accuracy"],
        optimizer="adam",
    )
    model.fit(X_train, y_train, epochs=10, batch_size=32)

    # Encoded classes
    print(list(enc.classes_))
    preds = model.predict(X_test)[0]
    for p in preds:
        m = np.argmax(p)
        print(enc.inverse_transform([m])[0])

    print(model.evaluate(X_test, y_test, batch_size=32))
    # The two final numerical scores express the loss and accuracy of the classifier for the test data; the 1.0 
    # indicates a 100% accuracy score.


if __name__ == "__main__":
    run()
