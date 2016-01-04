import abc
import numpy as np
from scipy.special import expit

# http://stackoverflow.com/questions/3985619/how-to-calculate-a-logistic-sigmoid-function-in-python
def sigmoid(x):
    return expit(x)

def relu(x):
    return np.maximum(0, x)

def transfer_sigmoid(x):
    s = sigmoid(x)
    return s, s * (1 - s)

def transfer_relu(x):
    return relu(x), (x > 0).astype(int)

def softmax(x):
    e = np.exp(x - np.max(x))  # prevent overflow
    if e.ndim == 1:
        return e / np.sum(e, axis=0)
    return e / np.array([np.sum(e, axis=1)]).T

# https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/sandbox/rnn/costs/gradient_clipping.py#L47
def clip_grad(grad, threshold):
    abs_grad = np.sqrt(np.sum(grad * grad))
    if abs_grad > threshold:
        grad *= threshold / abs_grad
    return grad

def grad_changes_sigmoid(lr, grad):
    return lr * grad

# With gradient clipping
# "We use clipping with a cut-off threshold of 6 on the norm of the gradients"
# http://arxiv.org/pdf/1211.5063.pdf

# But here they say 1 is better
# http://t-satoshi.blogspot.nl/2015/06/implementing-recurrent-neural-net-using.html
def grad_changes_relu(lr, grad):
    return lr * clip_grad(grad, 1)

def hsm(vi, h, W):
    classifiers = zip(vi.path, vi.code)
    res = 0
    for step, code in classifiers:
        t = 1 if code == 1 else -1
        sig = sigmoid(t * W[:, step].T.dot(h))
        res += np.log(sig if sig != 0 else 1)
    return res

# http://t-satoshi.blogspot.nl/2015/06/implementing-recurrent-neural-net-using.html
def random(shape):
    return np.random.normal(loc=0.0, scale=0.001, shape)