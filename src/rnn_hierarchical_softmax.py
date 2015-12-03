#!/usr/bin/env python3
import numpy as np
from rnn_routine import *

import h_softmax


class RNNHSoftmax:
    def __init__(self, hidden_layer_size, vocab):
        # Initialize model parameters
        # Word dimensions is the size of our vocubulary
        self.N = len(vocab)
        self.H = hidden_layer_size
        self.vocab = vocab

        # Randomly initialize weights
        self.U = np.random.randn(self.H, self.N)
        self.W = np.random.randn(self.H, self.H)

        # In RNNHSoftmax the outer word representation will dissappear
        self.V = np.random.randn(self.N, self.H)

        # Initial state of the hidden layer
        self.ntime = 3
        self.s = np.zeros((self.ntime, self.H))
        self.deriv_s = np.zeros((self.ntime, self.H))

    # Hierarchical softmax doesn't have outer word representation
    def word_representation_inner(self, word_idx):
        return self.U[:, word_idx]

    def _sentence_log_likelihood(self, Xi):
        hX = np.zeros((len(Xi), self.H))
        for idx, xi in enumerate(Xi):
            hX[idx] = self.U[:,xi]

        h = sigmoid(hX[:-1] + self.s[1].dot(self.W))
        return -np.sum([h_softmax.hsm(self.vocab[value], h[index], self.V.T)
                        for index, value in enumerate(Xi[1:])])

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

            err_hidden[0] = np.zeros(self.H)

            # Path for the next word
            classifiers = zip(self.vocab[di].path, self.vocab[di].code)
            for step, code in classifiers:
                p = sigmoid(self.V[step,:].T.dot(self.s[0]))
                g = p - code
                err_hidden[0] += g * self.V[step,:]
                der = g * self.s[0]
                self.V[step,:] += lr * der

            for i in range(1, self.ntime - 1):
                err_hidden[i] = self.W.T.dot(err_hidden[i - 1]) * self.deriv_s[i]

            # The same trick. Instead of updating a whole matrix by err_hidden[0] dot x
            self.U[:, xi] += lr * err_hidden[0]
            self.W += lr * err_hidden.T.dot(self.s[1:])