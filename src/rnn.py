#!/usr/bin/env python3
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


# Input size == Vocabulary size
N = 100

# Hidden layer size
H = 20

# Weights
U = np.random.randn(H, N)
W = np.random.randn(H, H)
V = np.random.randn(N, H)

# Initial state of the hidden layer
s = np.zeros(H)


def predict(w):
    return softmax(V.dot(sigmoid(U.dot(w) + W.dot(s))))


# Input vector
w = np.zeros(N)
w[6] = 1

predict(w)
