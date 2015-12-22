import numpy as np
from scipy.special import expit

cimport numpy as np
cimport cython

DTYPE = np.float
ctypedef np.float_t DTYPE_t

# http://stackoverflow.com/questions/3985619/how-to-calculate-a-logistic-sigmoid-function-in-python

@cython.boundscheck(False)
def sigmoid_float(float x):
    return expit(x)

@cython.boundscheck(False)
def sigmoid_vec(np.ndarray[DTYPE_t, ndim=1] x):
    return expit(x)

@cython.boundscheck(False)
def sigmoid_mat(np.ndarray[DTYPE_t, ndim=2] x):
    return expit(x)

@cython.boundscheck(False)
def softmax(np.ndarray[DTYPE_t, ndim=1] x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

@cython.boundscheck(False)
def clip_grad(np.ndarray[DTYPE_t, ndim=2] grad, float threshold):
    cdef float abs_grad = np.sqrt(np.sum(grad * grad))
    if abs_grad > threshold:
        grad *= threshold / abs_grad
    return grad

@cython.boundscheck(False)
def hsm(vi, np.ndarray[DTYPE_t, ndim=1] h, np.ndarray[DTYPE_t, ndim=2] W):
    classifiers = zip(vi.path, vi.code)
    cdef float res = 0
    cdef int t
    cdef float sig
    for step, code in classifiers:
        t = 1 if code == 1 else -1
        sig = sigmoid_float(t * W[:, step].T.dot(h))
        res += np.log(sig if sig != 0 else 1)
    return res

@cython.boundscheck(False)
def hsm2(vi, np.ndarray[DTYPE_t, ndim=1] h, tag, np.ndarray[DTYPE_t, ndim=2] V, np.ndarray[DTYPE_t, ndim=2] G):
    classifiers = zip(vi.path, vi.code)
    cdef float res = 0
    cdef int t
    cdef float sig
    for step, code in classifiers:
        t = 1 if code == 1 else -1
        sig = sigmoid_float(t * (V[:, step].dot(h) + G[step, tag]))
        res += np.log(sig if sig != 0 else 1)
    return res
