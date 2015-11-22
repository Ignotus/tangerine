import numpy as np
from utils import create_context_windows

# Notation used from: word2vec Parameter Learning Explained - Xin Rong
class SkipGram():

    def __init__(self, vocab_size, window_size, hidden_layer_size=20):

        # Safety checks
        assert window_size != 0
        assert (window_size % 2) == 0

        # Initialize model parameters
        self.V = vocab_size
        self.N = hidden_layer_size
        self.C = window_size

        # Randomly initialize weights
        self.W = np.random.randn(self.V, self.N)
        self.W_prime = np.random.randn(self.N, self.V)

    def train(self, sentence, eta=0.025, compute_ll=False):
        for center_word, context in create_context_windows(sentence, self.C):
            # x_k = np.zeros(self.V)
            # x_k[center_word] = 1
            h = self.W[center_word, :]
            u = np.dot(h, self.W_prime)
            
            # Softmax
            y = np.exp(u)
            y = y / np.sum(y)
        
            EI = 0

            for c in context:
                t = np.zeros(self.V)
                t[c] = 1
                e = y - t 
                EI += e

            # Update hidden->output matrix
            self.W_prime = self.W_prime - (eta * np.tile(EI, (self.N, 1)) *
                    np.tile(np.array([h]).transpose(), (1, self.V)))

            # Compute EH
            EH = np.zeros(self.N)
            for i in range(0, self.N):
                EH[i] = (np.sum(EI * self.W_prime[i, :]))

            # Update input->hidden matrix
            self.W[center_word, :] -= eta * EH

        # TODO ugly temp code for testing
        if compute_ll:
            ll = 0
            for center_word, context in create_context_windows(sentence, \
                    self.C):
                h = self.W[center_word, :]
                u = np.dot(h, self.W_prime)
                
                for c in context:
                    ll += u[c]
                
                ll -= self.C * np.log(np.sum(np.exp(u)))
                return ll



    def predict(self, x):
        pass
