import abc
import numpy as np
from scipy.special import expit

# http://stackoverflow.com/questions/3985619/how-to-calculate-a-logistic-sigmoid-function-in-python
def sigmoid(x):
    return expit(x)

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

def clip_grad(grad, threshold):
    abs_grad = np.sqrt(np.sum(grad * grad))
    if abs_grad > threshold:
        grad *= threshold / abs_grad
    return grad

def hsm(vi, h, W):
    classifiers = zip(vi.path, vi.code)
    res = 0
    for step, code in classifiers:
        t = 1 if code == 1 else -1
        sig = sigmoid(t * W[:, step].T.dot(h))
        res += np.log(sig if sig != 0 else 1)
    return res


def hsm2(vi, h, tag, V, G):
    classifiers = zip(vi.path, vi.code)
    res = 0
    for step, code in classifiers:
        t = 1 if code == 1 else -1
        sig = sigmoid(t * (V[:, step].dot(h) + G[step, tag]))
        res += np.log(sig if sig != 0 else 1)
    return res
