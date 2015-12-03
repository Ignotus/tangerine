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
        self.s = np.zeros((self.ntime, self.H))
        self.deriv_s = np.zeros((self.ntime, self.H))

    def word_representation_outer(self, word_idx):
        return self.V[word_idx, :]

    def word_representation_inner(self, word_idx):
        return self.U[:, word_idx]

    def predict(self, x):
        s_t = sigmoid(self.U.dot(x) + self.W.dot(self.s[1]))
        return np.argmax(softmax(self.V.dot(s_t)))

    def _sentence_log_likelihood(self, Xi):
        hX = np.zeros((len(Xi), self.H))
        for idx, xi in enumerate(Xi):
            hX[idx] = self.U[:,xi]

        h = sigmoid(hX[:-1] + self.s[1].dot(self.W))
        log_q = h.dot(self.V.T)
        a = np.max(log_q, axis=1)
        log_Z = a + np.log(np.sum(np.exp((log_q.T - a).T), axis=1))
        #print log_Z
        return np.sum(np.array([log_q[index, value]
                                for index, value in enumerate(Xi[1:])])
                      - log_Z)

    def log_likelihood(self, Xii):
        """
            Xii is a list of list of indexes. Each list represent separate sentence
        """
        return sum([self._sentence_log_likelihood(Xi) for Xi in Xii])

    def train(self, Xi, lr=0.1):
        err_hidden = np.empty((self.ntime - 1, self.H))
        for xi, di in zip(Xi, Xi[1:]):
            self.s[1:] = self.s[:-1]
            self.deriv_s[1:] = self.deriv_s[:-1]

            # self.U[:,xi] == self.U.dot(x) if x is one-hot-vector
            self.s[0] = sigmoid(self.U[:,xi] + self.W.dot(self.s[1]))
            self.deriv_s[0] = self.s[0] * (1 - self.s[0])

            err_out = -softmax(self.V.dot(self.s[0]))
            err_out[di] += 1

            self.V += lr * err_out[None].T.dot(self.s[0][None])

            err_hidden[0] = self.V.T.dot(err_out) * self.deriv_s[0]
            for i in range(1, self.ntime - 1):
                err_hidden[i] = self.W.T.dot(err_hidden[i - 1]) * self.deriv_s[i]

            # The same trick. Instead of updating a whole matrix by err_hidden[0] dot x
            self.U[:, xi] += lr * err_hidden[0]
            self.W += lr * err_hidden.T.dot(self.s[1:])
