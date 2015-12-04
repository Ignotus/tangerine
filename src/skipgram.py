import numpy as np
from enum import Enum
from utils import create_context_windows

class SkipGramOptimizations(Enum):
    none                 = 1
    negative_sampling    = 2
    hierarchical_softmax = 3

# Notation used from: word2vec Parameter Learning Explained - Xin Rong
class SkipGram():

    def __init__(self, vocab_size, optimizations=SkipGramOptimizations.none, \
                window_size=4, hidden_layer_size=100):

        # Set the correct training and log-likelihood functions
        if optimizations is SkipGramOptimizations.none:
            self.train_fun = self.__train_plain
            self.compute_LL_fun = self.__compute_LL_plain
        elif optimizations is SkipGramOptimizations.negative_sampling:
            pass
        elif optimizations is SkipGramOptimizations.hierarchical_softmax:
            pass

        # Initialize model parameters
        self.V = vocab_size
        self.N = hidden_layer_size
        self.C = window_size

        # Randomly initialize weights
        self.W = np.random.randn(self.V, self.N)
        self.W_prime = np.random.randn(self.N, self.V)

    def train(self, sentence, learning_rate=0.025):
        self.train_fun(sentence, learning_rate)

    def compute_LL(self, sentences):
        self.compute_LL_fun(sentences)

    def __train_plain(self, sentence, learning_rate):
        eta = learning_rate
        for center_word, context in create_context_windows(sentence, self.C): 

            # Retrieve the input vector for the center word
            h = self.W[center_word, :]

            # Calculate the net input scores for all words ([u_1, ..., u_V])
            u = np.dot(h, self.W_prime)

            # Calculate the probability distribution over all words using the
            # softmax
            y = np.exp(u)
            y = y / np.sum(y)
        
            # Calculate the summed prediction error for all context words
            EI = 0
            for c in context:
                t = np.zeros(self.V)
                t[c] = 1
                e = (self.C * y) - t 
                EI += e

            # Update hidden->output matrix
            self.W_prime = self.W_prime - (eta * np.tile(EI, (self.N, 1)) *
                    np.tile(np.array([h]).transpose(), (1, self.V)))

            # Compute EH
            EH = np.dot(EI, self.W_prime.transpose())

            # Update input->hidden matrix
            self.W[center_word, :] -= eta * EH

    def __compute_LL_plain(self, sentences):
        LL = 0
        for sentence in sentences:
            for center_word, context in create_context_windows(sentence, \
                    self.C):

                # Retrieve the input vector for the center word
                h = self.W[center_word, :]

                # Calculate the net input scores for all words ([u_1, ..., u_V])
                u = np.dot(h, self.W_prime)

                # Add the first term of the log-likelihood
                for c in context:
                    LL += u[c]

                # Add the second term of the log-likelihood
                LL -= self.C * np.log(np.sum(np.exp(u)))

        return LL

