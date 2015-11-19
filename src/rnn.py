#!/usr/bin/env python3
import numpy as np
from rnn_routine import *


class RNN:
    def __init__(self, word_dim, hidden_layer_size=20):
        # Initialize model parameters
        # Word dimensions is the size of our vocubulary
        self.N = word_dim
        self.H = hidden_layer_size

        # Randomly initialize weights
        self.U = np.random.randn(self.H, self.N)
        self.W = np.random.randn(self.H, self.H)
        self.V = np.random.randn(self.N, self.H)

        # Initial state of the hidden layer
        self.ntime = 3
        self.s = [np.zeros(self.H) for i in range(self.ntime)]

    def predict(self, x):
        s_t = sigmoid(self.U.dot(x) + self.W.dot(self.s))
        return softmax(self.V.dot(s_t))

    def train(self, Xi, lr=0.1):
        for xi, di in zip(Xi, Xi[1:]):
            x = np.zeros(self.N)
            x[xi] = 1
            d = np.zeros(self.N)
            d[di] = 1
            self.s[1:] = self.s[:-1]
            self.s[0] = sigmoid(self.U.dot(x) + self.W.dot(self.s[1]))

            y = softmax(self.V.dot(self.s[0]))
            err_out = d - y

            self.V += lr * err_out[np.newaxis].T.dot(self.s[0][np.newaxis])

            err_hidden = err_out[np.newaxis].dot(self.V).dot(self.s[0]) * (1 - self.s[0])

            self.U += lr * self.W.dot(err_hidden[np.newaxis].T)
            self.W += lr * self.s[1].dot(err_hidden.T)

            for i in range(1, self.ntime - 1):
                err_hidden = err_hidden[np.newaxis].dot(self.W).dot(self.s[i]) * (1 - self.s[i])
                self.W += lr * self.s[i + 1].dot(err_hidden.T)


