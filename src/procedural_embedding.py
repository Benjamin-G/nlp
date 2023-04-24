import matplotlib.pyplot as plt
import numpy as np
from keras import Sequential
from keras.layers import Embedding
from keras.utils import pad_sequences
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE


def tsne_plot(model, max_words=100):
    labels = []
    tokens = []

    n = 0
    for word in model:
        if n < max_words:
            tokens.append(model[word])
            labels.append(word)
            n += 1

    tsne_model = TSNE(
        perplexity=40,
        n_components=2,
        init="pca",
        n_iter=10000,
        random_state=23,
    )

    new_values = tsne_model.fit_transform(np.array(tokens))

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(8, 8))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords="offset points",
                     ha="right",
                     va="bottom")
    plt.show()


def run():
    docs = [
        "Chuck Berry rolled over everyone who came before him ? and turned up everyone who came after.We'll miss you",
        "Help protect the progress we've made in helping millions ofAmericans get covered.",
        "Let's leave our children and grandchildren a planet that's healthier than the one we have today.",
        "The American people are waiting for Senate leaders to do their jobs.",
        "We must take bold steps now ? climate change is already impacting millions of people.",
        "Don't forget to watch Larry King tonight",
        "Ivanka is now on Twitter - You can follow her",
        "Last night Melania and I attended the Skating with the Stars Gala at Wollman Rink in Central Park",
        "People who have the ability to work should. But with the government happy to send checks",
        "I will be signing copies of my new book",
    ]

    docs = [d.lower() for d in docs]

    count_vect = CountVectorizer().fit(docs)
    tokenizer = count_vect.build_tokenizer()

    input_array = []
    for doc in docs:
        x = []
        for token in tokenizer(doc):
            x.append(count_vect.vocabulary_.get(token))
        input_array.append(x)

    max_len = max([len(d) for d in input_array])

    input_array = pad_sequences(
        input_array,
        maxlen=max_len,
        padding="post",
    )

    model = Sequential()
    model.add(Embedding(100, 8, input_length=len(input_array[0])))
    # input_array = np.random.randint(100, size=(10, 10))
    model.compile("rmsprop", "mse")
    output_array = model.predict(input_array)

    M = {}
    for i in range(len(input_array)):
        for j in range(len(input_array[i])):
            M[input_array[i][j]] = output_array[i][j]

    tsne_plot(M)


if __name__ == "__main__":
    run()
