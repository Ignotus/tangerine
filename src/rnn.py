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
        self.grad_threshold = 100

    def word_representation(self, word_idx):
        return self.V[word_idx, :]

    def predict(self, x):
        s_t = sigmoid(self.U.dot(x) + self.W.dot(self.s[1]))
        return softmax(self.V.dot(self.s[0]))

    def sentence_log_likelihood(self, Xi):
        X = np.zeros((len(Xi), self.N))
        for idx, xi in enumerate(Xi):
            X[idx][xi] = 1

        h = X[:-1].dot(self.U.T) + self.s[1].dot(self.W)
        log_q = h.dot(self.V.T)
        a = np.max(log_q, axis=1)
        log_Z = a + np.log(np.sum(np.exp((log_q.T - a).T)))
        #print log_Z
        return np.sum(np.array([log_q[index, value]
                                for (index,), value in np.ndenumerate(Xi[1:])])
                      - log_Z)

    def log_likelihood(self, Xii):
        """
            Xii is a list of list of indexes. Each list represent separate sentence
        """
        return sum([self.sentence_log_likelihood(Xi) for Xi in Xii])

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

            self.V += lr * clip_grad(err_out[np.newaxis].T.dot(self.s[0][np.newaxis]),
                                     self.grad_threshold)

            err_hidden = err_out[np.newaxis].dot(self.V).dot(self.s[0]) * (1 - self.s[0])

            self.U += lr * clip_grad(err_hidden[np.newaxis].T.dot(x[np.newaxis]),
                                     self.grad_threshold)
            self.W += lr * clip_grad(self.s[1].dot(err_hidden.T),
                                    self.grad_threshold)

            for i in range(1, self.ntime - 1):
                err_hidden = err_hidden[np.newaxis].dot(self.W).dot(self.s[i]) * (1 - self.s[i])
                self.W += lr * clip_grad(self.s[i + 1].dot(err_hidden.T), self.grad_threshold)
