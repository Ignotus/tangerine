import numpy as np
from numpy.random import choice

class UnigramDistribution():
    
    def __init__(self, vocab, power):
        counts = np.array([count for _, count in vocab][: -1])
        counts = np.power(counts, power)
        Z = np.sum(counts)
        self.probs = counts / Z

    def sample(self, size):
        return choice(len(self.probs), size, p=self.probs)