#!/usr/bin/env python3
import numpy as np
from rnn_hierarchical_softmax import RNNHSoftmax
from rnn_routine import *

from spacy.parts_of_speech import ADJ, SPACE

NUM_OF_TAGS = SPACE - ADJ + 2


class RNNHSoftmaxPOS(RNNHSoftmax):
    def __init__(self, hidden_layer_size, vocab):
        super(RNNHSoftmaxPOS, self).__init__(hidden_layer_size, vocab)

        # COLING 2014 Tutorial-fix - Tomas Mikolov. Page 124
        self.F = np.random.randn(self.H, NUM_OF_TAGS)
        self.G = np.random.randn(self.N, NUM_OF_TAGS)

    def export(self, file_path):
        np.savez(file_path, self.N, self.H, self.U, self.W, self.V, self.s, self.deriv_s, self.F, self.G)

    def _sentence_log_likelihood(self, Xi):
        hX = np.zeros((len(Xi), self.H))
        for idx, (tag, xi) in enumerate(Xi):
            hX[idx] = self.U[:,xi] + self.F[:,tag]

        h = sigmoid(hX[:-1]) # Just don't use hidden layers + self.s[1].dot(self.W))
        return np.sum([hsm2(self.vocab[value], h[index], tag, self.V.T, self.G)
                       for index, (tag, value) in enumerate(Xi[1:])])

    def train(self, Xi, lr=0.1):
        err_hidden = np.empty((self.ntime - 1, self.H))
        for (tagx, xi), (tagd, di) in zip(Xi, Xi[1:]):
            self.s[1:] = self.s[:-1]
            self.deriv_s[1:] = self.deriv_s[:-1]

            # self.U[:,xi] == self.U.dot(x) if x is one-hot-vector
            self.s[0] = sigmoid(self.U[:,xi] + self.F[:,tagx] + self.W.dot(self.s[1]))
            self.deriv_s[0] = self.s[0] * (1 - self.s[0])

            err_hidden[0] = np.zeros(self.H)

            # Path for the next word
            classifiers = zip(self.vocab[di].path, self.vocab[di].code)
            for step, code in classifiers:
                p = sigmoid(self.V[step].dot(self.s[0]) + self.G[step,tagx])
                g = code - p
                err_hidden[0] += g * self.V[step] * self.deriv_s[0]
                der = g * self.s[0]
                self.V[step] += lr * der
                self.G[step,tagx] += lr * g

            for i in range(1, self.ntime - 1):
                err_hidden[i] = self.W.T.dot(err_hidden[i - 1]) * self.deriv_s[i]

            # The same trick. Instead of updating a whole matrix by err_hidden[0] dot x
            self.U[:, xi] += lr * err_hidden[0]
            self.W += lr * err_hidden.T.dot(self.s[1:])
            self.F[:, tagx] += lr * err_hidden[0]