#!/usr/bin/env python3
import numpy as np
from rnn_routine import *


class RNNHSoftmax:
    def __init__(self, hidden_layer_size, vocab, use_relu=False):
        # Initialize model parameters
        # Word dimensions is the size of our vocubulary
        print('Enabling ReLU:', use_relu)
        self.N = len(vocab)
        self.H = hidden_layer_size
        self.vocab = vocab
        self.use_relu = use_relu

        # Randomly initialize weights
        self.U = np.random.randn(self.H, self.N)
        if self.use_relu:
            self.W = np.identity(self.H)
        else:
            self.W = np.random.randn(self.H, self.H)

        # In RNNHSoftmax the outer word representation will dissappear
        self.V = np.random.randn(self.N, self.H)

        # Initial state of the hidden layer
        self.ntime = 5
        self.s = np.zeros((self.ntime, self.H))
        self.deriv_s = np.zeros((self.ntime, self.H))

        self.transfer_func = transfer_sigmoid if not use_relu else transfer_relu
        self.grad_changes = grad_changes_sigmoid if not use_relu else grad_changes_relu

    def export(self, file_path):
        np.savez(file_path, self.N, self.H, self.U, self.W, self.V, self.s, self.deriv_s)

    # Hierarchical softmax doesn't have outer word representation
    def word_representation_inner(self, word_idx):
        return self.U[:, word_idx]

    def _sentence_log_likelihood(self, Xi):
        propagation_func = sigmoid if not self.use_relu else relu
        prev_s = np.zeros(self.H)
        log_ll = 0
        for xi, di in zip(Xi, Xi[1:]):
            h = propagation_func(self.U[:,xi] + self.W.dot(prev_s))
            log_ll += hsm(self.vocab[di], h, self.V.T)
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

            err_hidden[0] = np.zeros(self.H)

            # Path for the next word
            classifiers = zip(self.vocab[di].path, self.vocab[di].code)
            for step, code in classifiers:
                p = sigmoid(self.V[step].dot(self.s[0]))
                g = code - p
                err_hidden[0] += g * self.V[step] * self.deriv_s[0]
                der = g * self.s[0]
                self.V[step] += lr * der

            for i in range(1, self.ntime - 1):
                err_hidden[i] = self.W.T.dot(err_hidden[i - 1]) * self.deriv_s[i]

            # The same trick. Instead of updating a whole matrix by err_hidden[0] dot x
            self.U[:, xi] += self.grad_changes(lr, err_hidden[0])
            self.W += self.grad_changes(lr, self.s[1:].T.dot(err_hidden))
