import re

import numpy as np
from keras import layers, Model, Input
from keras.backend import int_shape
from keras.layers import SimpleRNN, LSTM, Embedding, Activation, add, Permute, concatenate, Dense, Flatten, Multiply, \
    dot
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer


def create_tokenizer(training_data, test_data):
    """
    a lookup facility that converts words into numerical indices.
    This tokenizer is fitted on a vocabulary, and here, we use all words in both the training and test data.

    :param training_data:
    :type training_data:
    :param test_data:
    :type test_data:
    :return:
    :rtype:
    """
    max_story_len = 0
    max_query_len = 0
    with open(training_data) as f:
        text = []
        for line in f:
            m = re.match("^\d+\s([^\.]+)[\.].*", line.rstrip())
            if m:
                if len(m.group(1)) > max_query_len:
                    max_story_len = len(m.group(1).split(' '))
                text.append(m.group(1))
            else:
                m = re.match("^\d+\s([^\?]+)[\?]\s\t([^\t]+)", line.rstrip())
                if m:
                    if len(m.group(1)) > max_query_len:
                        max_query_len = len(m.group(1).split(' '))
                    text.append(m.group(1) + ' ' + m.group(2))

    with open(test_data) as f:
        for line in f:
            m = re.match("^\d+\s([^\.]+)[\.].*", line.rstrip())
            if m:
                if len(m.group(1)) > max_query_len:
                    max_story_len = len(m.group(1).split(' '))
                text.append(m.group(1))
            else:
                m = re.match("^\d+\s([^\?]+)[\?].*", line.rstrip())
                if m:
                    if len(m.group(1)) > max_query_len:
                        max_query_len = len(m.group(1).split(' '))
                    text.append(m.group(1))

    vocabulary = set([word for word in text])
    max_words = len(vocabulary)

    tokenizer = Tokenizer(num_words=max_words, char_level=False, split=' ')
    tokenizer.fit_on_texts(text)

    return tokenizer, max_words, max_story_len, max_query_len


def vectorize(s, tokenizer):
    vector = tokenizer.texts_to_sequences([s])
    return vector[0]


def process_stories(filename, tokenizer, max_story_len, max_query_len, vocab_size, use_context=False):
    with open(filename) as f:
        X = []
        Q = []
        y = []
        # n_questions = 0

        for line in f:
            m = re.match("^(\d+)\s(.+)\.", line.rstrip())
            if m:
                if int(m.group(1)) == 1:
                    story = {}
                story[int(m.group(1))] = m.group(2)
            else:
                m = re.match("^\d+\s(.+)\?\s\t([^\t]+)\t(.+)",
                             line.rstrip())
                if m:
                    question = m.group(1)
                    answer = m.group(2)
                    answer_ids = [int(x) for x in m.group(3).split(" ")]
                    if use_context == False:
                        facts = ' '.join([story[id] for id in answer_ids])
                        vectorized_fact = vectorize(facts, tokenizer)
                    else:
                        vectorized_fact = vectorize(' '.join(story.values()), tokenizer)

                    vectorized_question = vectorize(question, tokenizer)
                    vectorized_answer = vectorize(answer, tokenizer)

                    X.append(vectorized_fact)

                    Q.append(vectorized_question)

                    answer = np.zeros(vocab_size)
                    answer[vectorized_answer[0]] = 1
                    y.append(answer)

        X = pad_sequences(X, maxlen=max_story_len)
        Q = pad_sequences(Q, maxlen=max_query_len)

    return np.array(X), np.array(Q), np.array(y)


def process_stories_n_context(filename, tokenizer, vocab_size, use_context=0):
    with open(filename) as f:
        X = []
        Q = []
        y = []
        max_story_len = 0
        max_query_len = 0

        for line in f:
            m = re.match("^(\d+)\s(.+)\.", line.rstrip())
            if m:
                if int(m.group(1)) == 1:
                    story = {}
                story[int(m.group(1))] = m.group(2)
            else:
                m = re.match("^\d+\s(.+)\?\s\t([^\t]+)\t(.+)",
                             line.rstrip())
                if m:
                    question = m.group(1)
                    answer = m.group(2)
                    answer_ids = [int(x) for x in m.group(3).split(" ")]
                    facts = ' '.join([story[id] for id in answer_ids])
                    all_facts = ' '.join([story[id] for id in story])
                    facts_v = vectorize(facts, tokenizer)
                    all_facts_v = vectorize(all_facts, tokenizer)

                    if use_context == 0:
                        vectorized_fact = facts_v
                    elif use_context == -1:
                        vectorized_fact = all_facts_v
                    else:
                        x = min(use_context, len(story))

                        facts = ' '.join([story[id] for id in answer_ids]) + ' '
                        n = 0
                        for id in story:
                            if n < x and id not in answer_ids:
                                facts += story[id] + ' '
                                n += 1
                        vectorized_fact = vectorize(facts, tokenizer)
                    l = len(vectorized_fact)
                    if l > max_story_len:
                        max_story_len = l
                    vectorized_question = vectorize(question,
                                                    tokenizer)
                    l = len(vectorized_question)
                    if l > max_query_len:
                        max_query_len = l

                    vectorized_answer = vectorize(answer,
                                                  tokenizer)

                    X.append(vectorized_fact)
                    Q.append(vectorized_question)
                    answer = np.zeros(vocab_size)
                    answer[vectorized_answer[0]] = 1
                    y.append(answer)

    X = pad_sequences(X, maxlen=max_story_len)
    Q = pad_sequences(Q, maxlen=max_query_len)

    return np.array(X), np.array(Q), np.array(y), max_story_len, max_query_len


def create_model_rnn(trainingData, testData, context=False):
    """
     We will implement a branching model with two RNNs. These two RNNs handle the facts (stories) and the question.
    Their output is merged by concatenation and sent through a Dense layer that produces a scalar of the size of our
    answer vocabulary, consisting of probabilities. The model is seeded with answer vectors with one bit on (
    one-hot), so the highest probability in the output layer reflects the most probable bit, indicating a unique
    answer word in our lexicon.

    one for analyzing a story and one for analyzing a question
    :param trainingData:
    :type trainingData:
    :param testData:
    :type testData:
    :param context:
    :type context:
    :return:
    :rtype:
    """
    tokenizer, vocab_size, max_story_len, max_query_len = create_tokenizer(trainingData, testData)

    # X_tr, Q_tr, y_tr = process_stories(trainingData, tokenizer, max_story_len,
    # max_query_len, vocab_size, use_context=context)

    # X_te, Q_te, y_te = process_stories(testData, tokenizer, max_story_len,
    #                                    max_query_len, vocab_size, use_context=context)

    X_tr, Q_tr, y_tr, max_story_len_tr, max_query_len_tr = process_stories_n_context(trainingData, tokenizer,
                                                                                     vocab_size,
                                                                                     use_context=context)

    X_te, Q_te, y_te, max_story_len_te, max_query_len_te = process_stories_n_context(testData, tokenizer,
                                                                                     vocab_size, use_context=context)

    max_story_len = max(max_story_len_tr, max_story_len_te)
    max_query_len = max(max_query_len_tr, max_query_len_te)

    print('Vocab size:', vocab_size, 'unique words')
    print('Story max length:', max_story_len, 'words')
    print('Query max length:', max_query_len, 'words')

    embedding = layers.Embedding(vocab_size, 100)

    story = layers.Input(shape=(max_story_len,),
                         dtype='int32')
    encoded_story = embedding(story)
    encoded_story = SimpleRNN(30)(encoded_story)

    question = layers.Input(shape=(max_query_len,), dtype='int32')

    encoded_question = embedding(question)
    encoded_question = SimpleRNN(30)(encoded_question)

    merged = layers.concatenate([encoded_story, encoded_question])

    preds = layers.Dense(vocab_size, activation='softmax')(merged)

    model = Model([story, question], preds)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print(model.summary())

    return X_tr, Q_tr, y_tr, X_te, Q_te, y_te, model


def create_model_lstm(training_data, test_data, context):
    tokenizer, vocab_size, max_story_len, max_query_len = create_tokenizer(training_data, test_data)

    X_tr, Q_tr, y_tr, max_story_len_tr, max_query_len_tr = process_stories_n_context(training_data, tokenizer,
                                                                                     vocab_size,
                                                                                     use_context=context)

    X_te, Q_te, y_te, max_story_len_te, max_query_len_te = process_stories_n_context(test_data, tokenizer,
                                                                                     vocab_size, use_context=context)

    max_story_len = max(max_story_len_tr, max_story_len_te)
    max_query_len = max(max_query_len_tr, max_query_len_te)

    print('Vocab size:', vocab_size, 'unique words')
    print('Story max length:', max_story_len, 'words')
    print('Query max length:', max_query_len, 'words')

    embedding = layers.Embedding(vocab_size, 100)

    story = layers.Input(shape=(max_story_len,),
                         dtype='int32')
    encoded_story = embedding(story)
    encoded_story = LSTM(30)(encoded_story)

    question = layers.Input(shape=(max_query_len,),
                            dtype='int32')
    encoded_question = embedding(question)
    encoded_question = LSTM(30)(encoded_question)

    merged = layers.concatenate([encoded_story,
                                 encoded_question])

    preds = layers.Dense(vocab_size, activation='softmax')(merged)

    model = Model([story, question], preds)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print(model.summary())

    return X_tr, Q_tr, y_tr, X_te, Q_te, y_te, model


def create_model_mem_nn(training_data, test_data, context):
    tokenizer, vocab_size, max_story_len, max_query_len = create_tokenizer(training_data, test_data)

    X_tr, Q_tr, y_tr, max_story_len_tr, max_query_len_tr = process_stories_n_context(training_data, tokenizer,
                                                                                     vocab_size,
                                                                                     use_context=context)

    X_te, Q_te, y_te, max_story_len_te, max_query_len_te = process_stories_n_context(test_data, tokenizer,
                                                                                     vocab_size, use_context=context)

    max_story_len = max(max_story_len_tr, max_story_len_te)
    max_query_len = max(max_query_len_tr, max_query_len_te)

    print('Vocab size:', vocab_size, 'unique words')
    print('Story max length:', max_story_len, 'words')
    print('Query max length:', max_query_len, 'words')

    input = Input((max_story_len,))
    question = Input((max_query_len,))

    A = Embedding(input_dim=vocab_size,
                  output_dim=64)
    C = Embedding(input_dim=vocab_size,
                  output_dim=max_query_len)
    B = Embedding(input_dim=vocab_size,
                  output_dim=64,
                  input_length=max_query_len)

    input_A = A(input)
    input_C = C(input)
    question_B = B(question)

    input_question_match = dot([input_A, question_B],
                               axes=(2, 2))
    Probs = Activation('softmax')(input_question_match)

    O = add([Probs, input_C])
    O = Permute((2, 1))(O)

    final_match = concatenate([O, question_B])

    size = int_shape(final_match)[2]
    weights = Dense(size, activation='softmax')(final_match)

    merged = Multiply()([final_match, weights])
    answer = Flatten()(merged)

    answer = Dense(vocab_size)(answer)
    answer = Activation('softmax')(answer)

    model = Model([input, question], answer)
    model.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print(model.summary())

    return X_tr, Q_tr, y_tr, X_te, Q_te, y_te, model


def run_evaluate(training_data, test_data, create_model, context=False):
    X_tr, Q_tr, y_tr, X_te, Q_te, y_te, model = create_model(training_data, test_data, context)

    print('Training')
    model.fit([X_tr, Q_tr], y_tr,
              batch_size=32,
              epochs=10,
              verbose=1,
              validation_split=0.1)

    print('Evaluation')
    loss, acc = model.evaluate([X_te, Q_te], y_te, batch_size=32)

    print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))


def run():
    """
    :return:
    :rtype:
    """
    print('RNN')
    run_evaluate('../data/tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_train.txt',
                 '../data/tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_test.txt',
                 create_model_rnn,
                 context=False)

    print('\n\nLSTM')
    run_evaluate('../data/tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_train.txt',
                 '../data/tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_test.txt',
                 create_model_lstm,
                 context=False)

    print('\n\n End-to-end memory network')
    run_evaluate('../data/tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_train.txt',
                 '../data/tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_test.txt',
                 create_model_mem_nn,
                 context=False)


if __name__ == '__main__':
    run()
