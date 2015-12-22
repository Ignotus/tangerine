#!/usr/bin/env cython3

import numpy as np
from rnn_routine import *

cimport numpy as np
cimport cython

DTYPE = np.float
ctypedef np.float_t DTYPE_t


cdef class RNNHSoftmax:
    cdef int N
    cdef int H
    cdef int ntime
    cdef object vocab
    cdef np.ndarray U, W, V, s, deriv_s

    @cython.boundscheck(False)
    def __init__(self, int hidden_layer_size, vocab):
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

    def export(self, file_path):
        np.savez(file_path, self.N, self.H, self.U, self.W, self.V, self.s, self.deriv_s)

    # Hierarchical softmax doesn't have outer word representation
    @cython.boundscheck(False)
    def word_representation_inner(self, int word_idx):
        return self.U[:, word_idx]

    @cython.boundscheck(False)
    def _sentence_log_likelihood(self, Xi):
        cdef np.ndarray[DTYPE_t, ndim=2] hX = np.zeros((len(Xi), self.H))
        for idx, xi in enumerate(Xi):
            hX[idx] = self.U[:,xi]

        # Just don't use hidden layers + self.s[1].dot(self.W))
        cdef np.ndarray[DTYPE_t, ndim=2] h = sigmoid_mat(hX[:-1])
        return np.sum([hsm(self.vocab[value], h[index], self.V.T)
                       for index, value in enumerate(Xi[1:])])

    @cython.boundscheck(False)
    def log_likelihood(self, Xii):
        """
            Xii is a list of list of indexes. Each list represent separate sentence
        """
        return sum([self._sentence_log_likelihood(Xi) for Xi in Xii])

    @cython.boundscheck(False)
    def train(self, Xi, float lr=0.005):
        cdef np.ndarray[DTYPE_t, ndim=2] err_hidden = np.empty((self.ntime - 1, self.H))
        cdef float p
        cdef float g
        cdef np.ndarray[DTYPE_t, ndim=1] der

        for xi, di in zip(Xi, Xi[1:]):
            self.s[1:] = self.s[:-1]
            self.deriv_s[1:] = self.deriv_s[:-1]

            # self.U[:,xi] == self.U.dot(x) if x is one-hot-vector
            self.s[0] = sigmoid_vec(self.U[:,xi] + self.W.dot(self.s[1]))
            self.deriv_s[0] = self.s[0] * (1 - self.s[0])

            err_hidden[0] = np.zeros(self.H)

            # Path for the next word
            classifiers = zip(self.vocab[di].path, self.vocab[di].code)

            for step, code in classifiers:
                p = sigmoid_float(self.V[step].dot(self.s[0]))
                g = code - p
                err_hidden[0] += g * self.V[step] * self.deriv_s[0]
                der = g * self.s[0]
                self.V[step] += lr * der

            for i in range(1, self.ntime - 1):
                err_hidden[i] = self.W.T.dot(err_hidden[i - 1]) * self.deriv_s[i]

            # The same trick. Instead of updating a whole matrix by err_hidden[0] dot x
            self.U[:, xi] += lr * err_hidden[0]
            self.W += lr * err_hidden.T.dot(self.s[1:])
