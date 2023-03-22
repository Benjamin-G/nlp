from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import Perceptron

if __name__ == "__main__":
    # Load the data from these categories
    categories = ["alt.atheism", "sci.med"]
    train = fetch_20newsgroups(subset="train", categories=categories, shuffle=True)

    # Load the model and prep for training
    perceptron = Perceptron(max_iter=100)

    cv = CountVectorizer()
    X_train_counts = cv.fit_transform(train.data)

    tfidf_tf = TfidfTransformer()
    X_train_tfidf = tfidf_tf.fit_transform(X_train_counts)

    # Train the model
    # Train a simple perceptron on a vector representation of the documents in these
    # two classes. A vector is nothing more than a container (an ordered list of a
    # finite dimension) for numerical values.
    perceptron.fit(X_train_tfidf, train.target)

    # Test the model
    test_docs = ["Religion is widespread, even in modern times",
                 "His kidney failed", "The pope is a controversial leader",
                 "White blood cells fight off infections",
                 "The reverend had a heart attack in church"]

    # The vector representation is based on a statistical representation of words
    # called TF.IDF, which we discuss in section 1.3.2. For now, just assume TF.IDF is a
    # magic trick that turns documents into vectors that can be fed to a machine
    # learning algorithm.
    X_test_counts = cv.transform(test_docs)
    X_test_tfidf = tfidf_tf.transform(X_test_counts)

    # Use model to predict the category of the test docs
    pred = perceptron.predict(X_test_tfidf)

    for doc, category in zip(test_docs, pred):
        print(f"{doc!r} => {train.target_names[category]}")
    # Apparently, these few short texts can be linearly separated by a simple, weight-based
    # algorithm. This example is a huge simplification: the topics chosen are quite distinct.
    # In real life, linear algorithms fall short in separating topics that overlap and share similar
    # vocabulary.
