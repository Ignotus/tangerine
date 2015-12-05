#!/usr/bin/env python3
import numpy as np
from rnn_extended import RNNExtended
from rnn_routine import *

class RNNExtendedReLU(RNNExtended):
    def __init__(self, word_dim, hidden_layer_size=20, class_size=10000):
        super(RNNExtendedReLU, self).__init__(word_dim, hidden_layer_size, class_size)

        self.W = np.identity(self.H)
        self.grad_threshold = 0.01

    def predict(self, x):
        s_t = relu(self.U.dot(x) + self.W.dot(self.s[1]))
        return np.argmax(softmax(self.X.dot(s_t))) * self.class_size +\
                np.argmax(softmax(self.V.dot(s_t)))

    def _sentence_log_likelihood(self, Xi):
        hX = np.zeros((len(Xi), self.H))
        for idx, xi in enumerate(Xi):
            hX[idx] = self.U[:,xi]

        h = relu(hX[:-1] + self.s[1].dot(self.W))
        log_q = h.dot(self.V.T)
        a = np.max(log_q, axis=1)
        log_Z = a + np.log(np.sum(np.exp((log_q.T - a).T), axis=1))

        log_c = h.dot(self.X.T)
        a = np.max(log_c, axis=1)
        log_C = a + np.log(np.sum(np.exp((log_c.T - a).T), axis=1))

        #print log_Z
        return np.sum(np.array([log_q[index, value % self.class_size]
                                for index, value in enumerate(Xi[1:])])
                      - log_Z) +\
               np.sum(np.array([log_c[index, value // self.class_size]
                                for index, value in enumerate(Xi[1:])])
                      - log_C)

    def train(self, Xi, lr=0.1):
        err_hidden = np.empty((self.ntime - 1, self.H))
        for xi, di in zip(Xi, Xi[1:]):
            class_id = di // self.class_size

            self.s[1:] = self.s[:-1]
            #self.deriv_s[1:] = self.deriv_s[:-1]

            self.s[0] = relu(self.U[:,xi] + self.W.dot(self.s[1]))
            #self.deriv_s[0] = self.s[0] * (1 - self.s[0])

            err_out = -softmax(self.V.dot(self.s[0]))
            err_out[di % self.class_size] += 1

            err_c = -softmax(self.X.dot(self.s[0]))
            err_c[class_id] += 1

            self.V += lr * clip_grad(err_out[None].T.dot(self.s[0][None]), self.grad_threshold)
            self.X += lr * clip_grad(err_c[None].T.dot(self.s[0][None]), self.grad_threshold)

            err_hidden[0] = self.V.T.dot(err_out) + self.X.T.dot(err_c)
            for i in range(1, self.ntime - 1):
                err_hidden[i] = self.W.T.dot(err_hidden[i - 1])

            self.U[:, xi] += lr * clip_grad(err_hidden[0], self.grad_threshold)
            self.W += lr * clip_grad(err_hidden.T.dot(self.s[1:]), self.grad_threshold)
