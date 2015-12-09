import numpy as np
import os
import ada_grad
import ada_delta
from enum import Enum
from scipy.special import expit
from utils import create_context_windows, index2word_to_VocabItems
from h_softmax import encode_huffman
from unigram import UnigramDistribution

class SkipGramOptimizations(Enum):
    none                 = 1
    negative_sampling    = 2
    hierarchical_softmax = 3

class SampleMode(Enum):
    uniform = 1
    unigram = 2

class LRMode(Enum):
    normal    = 1
    ada_grad  = 2
    ada_delta = 3

# Notation used from: word2vec Parameter Learning Explained - Xin Rong
class SkipGram():

    def __init__(self, vocab_size, optimization=SkipGramOptimizations.none, \
                window_size=4, hidden_layer_size=100, vocab=None, \
                num_negative_samples=5, unigram_power=0.75, \
                sample_mode=SampleMode.unigram, learning_rate=0.025, \
                lr_mode=LRMode.normal, ada_delta_noise=1e-3, \
                ada_delta_decay_rate=0.95):

        # Set the correct training and log-likelihood functions
        self.optimization = optimization
        if optimization is SkipGramOptimizations.none:
            self.train_fun = self.__train_plain
            self.compute_LL_fun = self.__compute_LL_plain
        elif optimization is SkipGramOptimizations.negative_sampling:
            if vocab is None:
                raise Exception("Vocabulary is None.")

            self.sample_mode = sample_mode
            if sample_mode is SampleMode.unigram:
                self.unigram = UnigramDistribution(vocab, unigram_power)
            self.train_fun = self.__train_negative_sampling
            self.compute_LL_fun = self.__compute_LL_negative_sampling
        elif optimization is SkipGramOptimizations.hierarchical_softmax:
            if vocab is None:
                raise Exception("Vocabulary is None.")

            self.vocabulary = index2word_to_VocabItems(vocab)
            self.train_fun = self.__train_hierarchical_softmax
            self.compute_LL_fun = self.__compute_LL_hierarchical_softmax

        # Initialize model parameters
        self.V = vocab_size
        self.N = hidden_layer_size
        self.C = window_size
        self.K = num_negative_samples

        # Randomly initialize weights
        self.W = np.random.randn(self.V, self.N)
        self.W_prime = np.random.randn(self.N, self.V)

        # Set up everything for learning rate calculation
        self.eta = learning_rate
        self.lr_mode = lr_mode
        if lr_mode is LRMode.ada_grad:
            self.i_LR = ada_grad.LR(learning_rate, self.V, self.N)
            self.o_LR = ada_grad.LR(learning_rate, self.V, self.N)
        elif lr_mode is LRMode.ada_delta:
            self.i_LR = ada_delta.LR(self.V, self.N, noise=ada_delta_noise, \
                    decay_rate=ada_delta_decay_rate)
            self.o_LR = ada_delta.LR(self.V, self.N, noise=ada_delta_noise, \
                    decay_rate=ada_delta_decay_rate)

    def train(self, sentence):
        self.train_fun(sentence)

    def compute_LL(self, sentences):
        return self.compute_LL_fun(sentences)

    def store_word_vectors(self, words, location, name):
        filename = location + os.sep + name + "_input.txt"
        print("Storing input vector representations in " + filename)
        with open(filename, 'w') as output_file:
            for i, word in enumerate(words):
                vec = self.W[i, :]
                output_file.write(word[0] + " " + \
                        " ".join(str(f) for f in vec) + "\n")

        if self.optimization != SkipGramOptimizations.hierarchical_softmax:
            filename = location + os.sep + name + "_output.txt"
            print("Storing output vector representations in " + filename)
            with open(filename, 'w') as output_file:
                for i, word in enumerate(words):
                    vec = self.W_prime[:, i]
                    output_file.write(word[0] + " " + \
                            " ".join(str(f) for f in vec) + "\n")

    def __train_hierarchical_softmax(self, sentence):
        for center_word, context in create_context_windows(sentence, self.C): 
            EH = np.zeros(self.N)

            # Retrieve the input vector for the center word
            h = self.W[center_word, :]

            # Update the output matrix for every context word
            for context_word in context:
                path = zip(self.vocabulary[context_word].path,  
                        self.vocabulary[context_word].code)

                # Go through every inner node in the path to the context word
                for inner_node, next_left in path:
                    e = expit(np.dot(h, self.W_prime[:, inner_node])) -  \
                            next_left
                    EH += e * self.W_prime[:, inner_node]
                    grad = e * h

                    # Retrieve our learning rate
                    eta = self.__get_output_eta(inner_node, grad)

                    # Update the hidden->output matrix
                    self.W_prime[:, inner_node] -= eta * grad

            # Retrieve our learning rate
            eta = self.__get_input_eta(center_word, EH)

            # Update input->hidden matrix
            self.W[center_word, :] -= eta * EH

    def __compute_LL_hierarchical_softmax(self, sentences):
        LL = 0
        for sentence in sentences:
            for center_word, context in create_context_windows(sentence, \
                    self.C):

                # Retrieve the input vector for the center word
                h = self.W[center_word, :]

                # Sum the log probabilities for all context words
                for context_word in context:
                    path = zip(self.vocabulary[context_word].path,  
                            self.vocabulary[context_word].code)

                    # Go through every inner node in the path to the context
                    # word
                    for inner_node, next_left in path:
                        sign = 1 if bool(next_left) else -1
                        LL += np.log(expit(sign * np.dot(h, self.W_prime[:, 
                                inner_node])))

        return LL

    def __train_negative_sampling(self, sentence):
        for center_word, context in create_context_windows(sentence, self.C): 
            EH = np.zeros(self.N)

            # Retrieve the input vector for the center word
            h = self.W[center_word, :]

            # Update the hidden->output matrix for each context word
            for context_word in context:

                # Update the hidden->output matrix for the positive sample
                e = expit(np.dot(self.W_prime[:, context_word], h)) - 1.0
                EH += e * self.W_prime[:, context_word]
                grad = e * h
                eta = self.__get_output_eta(context_word, grad)
                self.W_prime[:, context_word] -= eta *  grad

                # Draw K negative samples
                negative_samples = self.__draw_negative_samples(self.K)

                # Update the hidden->output matrix for each negative sample
                for ns in negative_samples:
                    e = expit(np.dot(self.W_prime[:, ns], h))
                    EH += e * self.W_prime[:, ns]
                    grad = e * h
                    eta = self.__get_output_eta(ns, grad)
                    self.W_prime[:, ns] -= eta * grad

            # Get the learning rate
            eta = self.__get_input_eta(center_word, EH)

            # Update the input->hidden matrix
            self.W[center_word, :] -= eta * EH

    def __compute_LL_negative_sampling(self, sentences):
        LL = 0
        for sentence in sentences:
            for center_word, context in create_context_windows(sentence, \
                    self.C):

                # Retrieve the input vector for the center word
                h = self.W[center_word, :]

                # Sum the log probabilities for all context words
                for context_word in context:

                    # Update the log-likelihood for the positive sample
                    LL += np.log(expit(np.dot(self.W_prime[:, context_word], \
                            h)))

                    # Draw K negative samples
                    negative_samples = self.__draw_negative_samples(self.K)

                    # Update the log-likelihood for all the negative samples
                    for ns in negative_samples:
                        LL += np.log(expit(np.dot(-1.0 * self.W_prime[:, ns], \
                                h)))

        return LL

    def __train_plain(self, sentence):
        
        # Plain has no support for AdaGrad
        eta = self.eta

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
                e = (len(context) * y) - t 
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
                LL -= len(context) * np.log(np.sum(np.exp(u)))

        return LL

    def __draw_negative_samples(self, amount):
        if self.sample_mode is SampleMode.uniform:
            return np.random.randint(0, self.V, amount)
        elif self.sample_mode is SampleMode.unigram:
            return self.unigram.sample(amount)

    def __get_input_eta(self, index, grad):
        if self.lr_mode is LRMode.normal:
            return self.eta
        elif self.lr_mode is LRMode.ada_grad:
            eta = self.i_LR.getLR(index)
            self.i_LR.updateTotalGrad(index, grad)
            return eta
        else:
            return self.i_LR.get_and_update_LR(index, grad)


    def __get_output_eta(self, index, grad):
        if self.lr_mode is LRMode.normal:
            return self.eta
        elif self.lr_mode is LRMode.ada_grad:
            eta = self.o_LR.getLR(index)
            self.o_LR.updateTotalGrad(index, grad)
            return eta
        else:
            return self.o_LR.get_and_update_LR(index, grad)
