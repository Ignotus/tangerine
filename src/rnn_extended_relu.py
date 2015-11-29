#!/usr/bin/env python3
import numpy as np
import rnn
from rnn_routine import *

class RNNExtendedReLU:
    def __init__(self, word_dim, hidden_layer_size=20, class_size=10000):
        # Initialize model parameters
        # Word dimensions is the size of our vocubulary
        self.N = word_dim
        self.H = hidden_layer_size
        self.class_size = class_size

        # Randomly initialize weights
        self.U = np.random.randn(self.H, self.N)

        self.W = np.identity(self.H)
        self.V = np.random.randn(self.class_size, self.H)
        nclass = np.ceil(float(self.N) / self.class_size)
        print("Number of classes: %d" % (nclass))
        self.X = np.random.randn(nclass, self.H)

        # Initial state of the hidden layer
        self.ntime = 3
        self.s = [np.zeros(self.H) for i in range(self.ntime)]

        self.grad_threshold = 0.01

    def word_representation(self, word_idx):
        idx = word_idx % self.class_size
        word_class = word_idx // self.class_size
        return np.hstack((self.V[idx, :], self.X[word_class, :]))

    def predict(self, x):
        s_t = relu(self.U.dot(x) + self.W.dot(self.s[1]))
        return np.argmax(softmax(self.X.dot(s_t))) * self.class_size +\
                np.argmax(softmax(self.V.dot(s_t)))

    def _sentence_log_likelihood(self, Xi):
        X = np.zeros((len(Xi), self.N))
        for idx, xi in enumerate(Xi):
            X[idx][xi] = 1

        h = relu(X[:-1].dot(self.U.T) + self.s[1].dot(self.W))
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

    def log_likelihood(self, Xii):
        """
            Xii is a list of list of indexes. Each list represent separate sentence
        """
        return sum([self._sentence_log_likelihood(Xi) for Xi in Xii])

    def train(self, Xi, lr=0.1):
        for xi, di in zip(Xi, Xi[1:]):
            x = np.zeros(self.N)
            x[xi] = 1
            class_id = di // self.class_size

            self.s[1:] = self.s[:-1]
            self.s[0] = relu(self.U.dot(x) + self.W.dot(self.s[1]))

            err_out = -softmax(self.V.dot(self.s[0]))
            err_out[di % self.class_size] += 1

            err_c = -softmax(self.X.dot(self.s[0]))
            err_c[class_id] += 1

            # Use clip grad with ReLU to avoid overflow
            # https://stats.stackexchange.com/questions/108381/how-to-avoid-nan-in-using-relu-cross-entropy
            self.V += lr * clip_grad(err_out[np.newaxis].T.dot(self.s[0][np.newaxis]), self.grad_threshold)
            self.X += lr * clip_grad(err_c[np.newaxis].T.dot(self.s[0][np.newaxis]), self.grad_threshold)

            err_hidden = (err_c[np.newaxis].dot(self.X) + err_out[np.newaxis].dot(self.V)).flatten()

            self.U += lr * clip_grad(err_hidden[np.newaxis].T.dot(x[np.newaxis]), self.grad_threshold)
            self.W += lr * clip_grad(self.s[1].dot(err_hidden.T), self.grad_threshold)

            for i in range(1, self.ntime - 1):
                err_hidden = err_hidden[np.newaxis].dot(self.W)
                self.W += lr * clip_grad(self.s[i + 1].dot(err_hidden.T), self.grad_threshold)


