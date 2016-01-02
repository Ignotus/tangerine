#!/usr/bin/env python3
import numpy as np
import rnn
from rnn_routine import *

class RNNExtended:
    def __init__(self, word_dim, hidden_layer_size=20, class_size=10000, use_relu=False):
        # Initialize model parameters
        # Word dimensions is the size of our vocubulary
        print('Enabling ReLU:', use_relu)
        self.N = word_dim
        self.H = hidden_layer_size
        self.class_size = class_size
        self.use_relu = use_relu

        # Randomly initialize weights
        self.U = random((self.H, self.N))
        if self.use_relu:
            self.W = np.identity(self.H)
        else:
            self.W = random((self.H, self.H))
        self.V = random((self.class_size, self.H))
        nclass = np.ceil(float(self.N) / self.class_size)
        print("Number of classes: %d" % (nclass))
        self.X = random((nclass, self.H))

        # Initial state of the hidden layer
        self.ntime = 4
        self.s = np.zeros((self.ntime, self.H))
        self.deriv_s = np.zeros((self.ntime, self.H))

        self.transfer_func = transfer_sigmoid if not use_relu else transfer_relu
        self.grad_changes = grad_changes_sigmoid if not use_relu else grad_changes_relu

    def export(self, file_path):
        np.savez(file_path, self.N, self.H, self.U, self.W, self.V, self.X, self.s, self.deriv_s)

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

            log_ll += log_q[di % self.class_size] - log_Z

            log_c = self.X.dot(h)
            a = np.max(log_c)
            log_C = a + np.log(np.sum(np.exp(log_c - a)))

            log_ll += log_c[di // self.class_size] - log_C

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
            class_id = di // self.class_size

            self.s[1:] = self.s[:-1]
            self.deriv_s[1:] = self.deriv_s[:-1]

            self.s[0], self.deriv_s[0] = self.transfer_func(self.U[:,xi] + self.W.dot(self.s[1]))

            err_out = -softmax(self.V.dot(self.s[0]))
            err_out[di % self.class_size] += 1

            err_c = -softmax(self.X.dot(self.s[0]))
            err_c[class_id] += 1

            self.V += self.grad_changes(lr, err_out[None].T.dot(self.s[0][None]))
            self.X += self.grad_changes(lr, err_c[None].T.dot(self.s[0][None]))

            err_hidden[0] = (self.V.T.dot(err_out) + self.X.T.dot(err_c)) * self.deriv_s[0]
            for i in range(1, self.ntime - 1):
                err_hidden[i] = self.W.T.dot(err_hidden[i - 1]) * self.deriv_s[i]

            self.U[:, xi] += self.grad_changes(lr, err_hidden[0])
            self.W += self.grad_changes(lr, self.s[1:].T.dot(err_hidden))

