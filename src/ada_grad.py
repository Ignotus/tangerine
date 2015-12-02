import numpy as np

# the class can be instantiated and will contain learning rates for parameters
class LR:
    grads = None  # accumulated grads array for every parameter
    alpha = None  # the initial learning rate

    # x: the number of words
    # y: the dimensionality of wordvectors
    def __init__(self, alpha, x, y):
        self.alpha = alpha
        self.grads = []
        for i in range(x):
            self.grads.append(np.ones(y))

    # returns a vector of learning rates
    # i: the word i
    def getLR(self, i):
        return self.alpha / np.sqrt(self.grads[i])

    # i: is the row of parameters
    # grad: a vector of gradients
    def updateTotalGrad(self, i, grad):
        self.grads[i] += np.power(grad, 2)