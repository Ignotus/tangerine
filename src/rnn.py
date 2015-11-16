#!/usr/bin/env python3
from collections import defaultdict
import numpy as np
from nltk.tokenize import RegexpTokenizer


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


# Iterations
niter = 10

# Input size == Vocabulary size
N = 100

# Hidden layer size
H = 20

# Weights
U = np.random.randn(H, N)
W = np.random.randn(H, H)
V = np.random.randn(N, H)

# Initial state of the hidden layer
ntime = 3
s = [np.zeros(H) for i in range(ntime)]


def predict(x):
    s_t = sigmoid(U.dot(x) + W.dot(s))
    return softmax(V.dot(s_t))


def train(Xi, nwords, lr=0.1):
    global U, W, V, s, N
    N = nwords
    U = np.random.randn(H, N)
    V = np.random.randn(N, H)

    for xi, di in zip(Xi, Xi[1:]):
        x = np.zeros(N)
        x[xi] = 1
        d = np.zeros(N)
        d[di] = 1
        s[1:] = s[:-1]
        s[0] = sigmoid(U.dot(x) + W.dot(s[1]))

        y = softmax(V.dot(s[0]))
        err_out = d - y

        V += lr * err_out[np.newaxis].T.dot(s[0][np.newaxis])

        err_hidden = err_out[np.newaxis].dot(V).dot(s[0]) * (1 - s[0])

        U += lr * W.dot(err_hidden[np.newaxis].T)

        W += lr * s[1].dot(err_hidden.T)

        for i in range(1, ntime - 1):
            err_hidden = err_hidden[np.newaxis].dot(W).dot(s[i]) * (1 - s[i])
            W += lr * s[i + 1].dot(err_hidden.T)


if __name__ == '__main__':
    tokenizer = RegexpTokenizer(r'\w+')
    content = open('test.txt').read()
    sentences = list(filter(None,
                            [[token.lower()
                              for token in tokenizer.tokenize(sentence)]
                             for sentence in content.split('.')]))

    word_indexes = defaultdict(lambda: len(word_indexes))
    for sentence in sentences:
        for token in sentence:
            word_indexes[token]
    word_indexes['$']

    indexes = [[word_indexes[token] for token in sentence] + [word_indexes['$']]
               for sentence in sentences]

    for i in range(niter):
        for sentence_indexes in indexes:
            train(sentence_indexes, len(word_indexes))
