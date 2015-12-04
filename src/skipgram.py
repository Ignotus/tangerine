import numpy as np
from enum import Enum
from utils import create_context_windows
from scipy.special import expit

class SkipGramOptimizations(Enum):
    none                 = 1
    negative_sampling    = 2
    hierarchical_softmax = 3

# Notation used from: word2vec Parameter Learning Explained - Xin Rong
class SkipGram():

    def __init__(self, vocab_size, optimizations=SkipGramOptimizations.none, \
                window_size=4, hidden_layer_size=100, num_negative_samples=10):

        # Set the correct training and log-likelihood functions
        if optimizations is SkipGramOptimizations.none:
            self.train_fun = self.__train_plain
            self.compute_LL_fun = self.__compute_LL_plain
        elif optimizations is SkipGramOptimizations.negative_sampling:
            self.train_fun = self.__train_negative_sampling
            self.compute_LL_fun = self.__compute_LL_negative_sampling
        elif optimizations is SkipGramOptimizations.hierarchical_softmax:
            pass

        # Initialize model parameters
        self.V = vocab_size
        self.N = hidden_layer_size
        self.C = window_size
        self.K = num_negative_samples

        # Randomly initialize weights
        self.W = np.random.randn(self.V, self.N)
        self.W_prime = np.random.randn(self.N, self.V)

    def train(self, sentence, learning_rate=0.025):
        self.train_fun(sentence, learning_rate)

    def compute_LL(self, sentences):
        return self.compute_LL_fun(sentences)

    def __train_negative_sampling(self, sentence, eta):
        for center_word, context in create_context_windows(sentence, self.C): 

            # Retrieve the input vector for the center word
            h = self.W[center_word, :]

            # Draw K negative samples (TODO unigram distribution)
            negative_samples = np.random.randint(0, self.V, self.K)
            positive_sample = center_word

            # Calculate the net input scores for all samples ([u_1, ..., u_K+1])
            u = np.zeros(self.K + 1)
            for j in range(self.K):
                u[j] = np.dot(h, self.W_prime[:, negative_samples[j]])
            u[self.K] = np.dot(h, self.W_prime[:, positive_sample])

            # Calculate the output values for all samples
            y = expit(u)

            # Calculate the summed prediction errors
            EI = np.zeros(self.K + 1)
            for j in range(self.K):
                for c in context:
                    EI[j] += (y[j] - (c == negative_samples[j]))

            # Calculate the summed prediction error for the positive sample
            for c in context:
                EI[self.K] += (y[self.K] - (c == center_word))

            # Update the hidden->ouput matrix for the negative samples
            for j in range(self.K):
                pass


    def __compute_LL_negative_sampling(self, sentences):
        pass

    def __train_plain(self, sentence, eta):
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

