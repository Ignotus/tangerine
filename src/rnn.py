#!/usr/bin/env python3
import numpy as np
from rnn_routine import *


class RNN:
    def __init__(self, word_dim, hidden_layer_size=20, use_relu=False):
        # Initialize model parameters
        # Word dimensions is the size of our vocubulary
        print('Enabling ReLU:', use_relu)
        self.N = word_dim
        self.H = hidden_layer_size
        self.use_relu = use_relu

        # Randomly initialize weights
        self.U = np.random.randn(self.H, self.N)
        if self.use_relu:
            self.W = np.identity(self.H)
        else:
            self.W = np.random.randn(self.H, self.H)
        self.V = np.random.randn(self.N, self.H)


        # Initial state of the hidden layer
        self.ntime = 5
        self.s = np.zeros((self.ntime, self.H))
        self.deriv_s = np.zeros((self.ntime, self.H))

        self.transfer_func = transfer_sigmoid if not use_relu else transfer_relu
        self.grad_changes = grad_changes_sigmoid if not use_relu else grad_changes_relu

    def export(self, file_path):
        np.savez(file_path, self.N, self.H, self.U, self.W, self.V, self.s, self.deriv_s)

    def load(self, file_path):
        npzfile = np.load(file_path)
        self.N = npzfile['arr_0']
        self.H = npzfile['arr_1']
        self.U = npzfile['arr_2']
        self.W = npzfile['arr_3']
        self.V = npzfile['arr_4']

    def word_representation_outer(self, word_idx):
        return self.V[word_idx, :]

    def word_representation_inner(self, word_idx):
        return self.U[:, word_idx]

    def _sentence_log_likelihood(self, Xi):
        propagation_func = sigmoid if not self.use_relu else relu
        prev_s = np.zeros(self.H)
        log_ll = 0
        for xi, di in zip(Xi, Xi[1:]):
            h = propagation_func(self.U[:,xi] + self.W.dot(prev_s))
            log_q = self.V.dot(h)
            a = np.max(log_q)
            log_Z = a + np.log(np.sum(np.exp(log_q - a)))

            log_ll += log_q[di] - log_Z
            prev_s = h

        return log_ll

    def log_likelihood(self, Xii):
        """
            Xii is a list of list of indexes. Each list represent separate sentence
        """
        return sum([self._sentence_log_likelihood(Xi) for Xi in Xii])


    def train(self, Xi, lr=0.005):
        err_hidden = np.empty((self.ntime - 1, self.H))
        for xi, di in zip(Xi, Xi[1:]):
            self.s[1:] = self.s[:-1]
            self.deriv_s[1:] = self.deriv_s[:-1]

            # self.U[:,xi] == self.U.dot(x) if x is one-hot-vector
            self.s[0], self.deriv_s[0] = self.transfer_func(self.U[:,xi] + self.W.dot(self.s[1]))

            err_out = -softmax(self.V.dot(self.s[0]))
            err_out[di] += 1

            self.V += self.grad_changes(lr, err_out[None].T.dot(self.s[0][None]))

            err_hidden[0] = self.V.T.dot(err_out) * self.deriv_s[0]
            for i in range(1, self.ntime - 1):
                err_hidden[i] = self.W.T.dot(err_hidden[i - 1]) * self.deriv_s[i]

            # The same trick. Instead of updating a whole matrix by err_hidden[0] dot x
            self.U[:, xi] += self.grad_changes(lr, err_hidden[0])
            # Each column of a matrix W should be influenced by separate err_hidden elements
            # becauses s[0] is influenced by W.dot(self.s[1])
            self.W += self.grad_changes(lr, self.s[1:].T.dot(err_hidden))
