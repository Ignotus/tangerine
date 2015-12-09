import numpy as np

class LR:

    def __init__(self, vocab_size, word_dim, decay_rate=0.95, noise=1e-3):
        self.rho = decay_rate
        self.epsilon = noise
        self.Eg = np.zeros((vocab_size, word_dim))
        self.Edx = np.zeros((vocab_size, word_dim))

    def get_and_update_LR(self, index, gradient):
        self.Eg[index] = (self.rho * self.Eg[index]) + \
                ((1.0 - self.rho) * np.power(gradient, 2))
        lr = (np.sqrt(self.Edx[index] + self.epsilon)) / \
                (np.sqrt(self.Eg[index] + self.epsilon))        
        dx = -lr * gradient
        self.Edx[index] = (self.rho * self.Edx[index]) + \
                ((1.0 - self.rho) * np.power(dx, 2))
        return lr