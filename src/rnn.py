#!/usr/bin/env python3
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


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


def train(X, D, lr=0.1):
    global U, W, V, s
    for x, d in zip(X, D):
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


# Input vector
X = np.zeros((2, N))
X[0][6] = 1
X[1][3] = 1

D = X

train(X, D)
