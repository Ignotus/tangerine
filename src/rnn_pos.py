#!/usr/bin/env python3
import numpy as np
from rnn import RNN
from rnn_routine import *

from spacy.parts_of_speech import ADJ, SPACE

NUM_OF_TAGS = SPACE - ADJ + 2

class RNNPOS(RNN):
    def __init__(self, word_dim, hidden_layer_size=20):
        super(RNNPOS, self).__init__(word_dim, hidden_layer_size)

        # COLING 2014 Tutorial-fix - Tomas Mikolov. Page 124
        self.F = np.random.randn(self.H, NUM_OF_TAGS)
        self.G = np.random.randn(self.N, NUM_OF_TAGS)

    def export(self, file_path):
        np.savez(file_path, self.N, self.H, self.U, self.W, self.V, self.s, self.deriv_s, self.F, self.G)

    def _sentence_log_likelihood(self, Xi):
        hX = np.zeros((len(Xi), self.H))
        hTags = np.zeros((len(Xi), NUM_OF_TAGS))
        for idx, (tag, xi) in enumerate(Xi):
            hX[idx] = self.U[:,xi] + self.F[:,tag]
            hTags[idx][tag] = 1

        h = sigmoid(hX[:-1]) # Just don't use hidden layers + self.s[1].dot(self.W))
        log_q = h.dot(self.V.T) + hTags[:-1].dot(self.G.T)
        a = np.max(log_q, axis=1)
        log_Z = a + np.log(np.sum(np.exp((log_q.T - a).T), axis=1))
        #print log_Z
        return np.sum(np.array([log_q[index, value]
                                for index, (tag, value) in enumerate(Xi[1:])])
                      - log_Z)

    def train(self, Xi, lr=0.1):
        err_hidden = np.empty((self.ntime - 1, self.H))
        for (tagx, xi), (tagd, di) in zip(Xi, Xi[1:]):
            self.s[1:] = self.s[:-1]
            self.deriv_s[1:] = self.deriv_s[:-1]

            # self.U[:,xi] == self.U.dot(x) if x is one-hot-vector
            self.s[0] = sigmoid(self.U[:,xi] + self.F[:,tagx] + self.W.dot(self.s[1]))
            self.deriv_s[0] = self.s[0] * (1 - self.s[0])

            err_out = -softmax(self.V.dot(self.s[0]) + self.G[:,tagx])
            err_out[di] += 1

            self.V += lr * err_out[None].T.dot(self.s[0][None])
            self.G[:,tagx] += lr * err_out

            err_hidden[0] = self.V.T.dot(err_out) * self.deriv_s[0]
            for i in range(1, self.ntime - 1):
                err_hidden[i] = self.W.T.dot(err_hidden[i - 1]) * self.deriv_s[i]

            # The same trick. Instead of updating a whole matrix by err_hidden[0] dot x
            self.U[:, xi] += lr * err_hidden[0]
            self.W += lr * err_hidden.T.dot(self.s[1:])
            self.F[:, tagx] += lr * err_hidden[0]
