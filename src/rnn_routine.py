import numpy as np
from scipy.special import expit

# http://stackoverflow.com/questions/3985619/how-to-calculate-a-logistic-sigmoid-function-in-python
def sigmoid(x):
    return expit(x)

def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)