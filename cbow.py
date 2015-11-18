import numpy as np
from sys import argv
import matplotlib.pyplot as plt
import nltk.data
from nltk.tokenize import RegexpTokenizer
from random import randint


class CBOW:
    V = None  # input matrix
    W = None  # output matrix
    vocab = None
    text = None

    m = None  # the number of sentences
    v = None  # the size of the vocab.
    n = None  # the number of word vector components.

    C = None  # the size of the context window



    # n the number of dimensions of vectors
    def __init__(self, file, alpha, epochs, C, n):
        self.text, self.vocab = self.readFile(file)
        self.m = len(self.text)
        self.v = len(self.vocab)
        self.n = n
        self.C = C
        self.V = np.ones((self.v, self.n))
        self.W = np.ones((self.n, self.v))
        self.train(alpha, epochs)


    def train(self, alpha, iter):
        error = []
        ep = []

        # starting LL
        error.append(self.computeLL())
        ep.append(0)

        for e in range(1, iter + 1):


            if e % 100 == 0:
                error.append(self.computeLL())
                ep.append(e)
                print('-------- ITER ' + str(e) + ' ---------')

            s = randint(0, self.m-1)  # pick sentence randomly
            l = len(self.text[s])
            i = randint(self.C, l - self.C-1)  # pick a center word randomly

            t = self.vocab[self.text[s][i]]  # the actual center word id

            h = np.zeros(self.v)
            # over context to compute h
            for j in range(1, self.C + 1):
                h[self.vocab[self.text[s][i + j]]] += 1
                h[self.vocab[self.text[s][i - j]]] += 1

            # final computation of h
            h = (1 / self.C) * self.V.T.dot(h)

            # over context again now to update weights
            for j in range(1, self.C + 1):
                w1 = self.vocab[self.text[s][i + j]]  # one to the right
                w2 = self.vocab[self.text[s][i - j]]  # one to the left
                EH = self.getEH(t, h)

                # 1. update the input matrix
                self.V[w1, :] -= (alpha / self.C) * EH
                self.V[w2, :] -= (alpha / self.C) * EH

            for id in range(0, self.v):
                # 2. update the output matrix
                self.W[:, id] -= alpha * self.getE(id, t, h) * h

        print(error)
        plt.plot(ep, error)
        plt.show()

    ########## SUPPORT FUNCTIONS ###########

    # jth component will be one
    def getOneHotVector(self, j, dim):
        vec = np.zeros(dim)
        vec[j] = 1
        return vec

    def computeLL(self):
        TE = 0
        # iterate over sentences
        for s in range(0, self.m):
            l = len(self.text[s])
            for i in range(self.C, l - self.C):
                t = self.vocab[self.text[s][i]]  # the actual center word
                h = np.zeros(self.v)
                for j in range(1, self.C + 1):
                    h[self.vocab[self.text[s][i + j]]] += 1
                    h[self.vocab[self.text[s][i - j]]] += 1
                    # final computation of h
                h = (1 / self.C) * self.V.T.dot(h)
                TE += np.log(self.y(t, h))
        return TE

    def getEH(self, t, h):
        res = 0
        for j in range(0, self.v):
            res += self.getE(j, t, h) * (self.W[:, j])
        return res

    # h: hidden neuron values
    def getE(self, j, t, h):
        if (t == j):
            r = self.y(j, h) - 1
        else:
            r = self.y(j, h)
        return r

    # softmax
    def y(self, j, h):
        Z = 0
        for i in range(0, self.v):
            Z += np.exp(self.W[:, i].T.dot(h))
        if (Z == 0): return 0
        return np.exp(self.W[:, j].T.dot(h)) / Z


    # returns text and the vocabulary in the form of the hash
    def readFile(self, file):

        fp = open(file)
        data = fp.read()
        sent = nltk.sent_tokenize(data)

        tokenizer = RegexpTokenizer(r'\w+')
        text = []
        voc = {}
        i = 0
        for j in range(0, len(sent)):
            words = tokenizer.tokenize(sent[j])
            text.append([])
            for w in words:
                w = w.lower()
                text[j].append(w)
                if not w in voc:
                    voc[w] = i
                    i += 1

        print('--------FINISHED READING FILE---------')
        return text, voc

        # txt = open(file)
        # text = txt.read().split(" ")
        # un = np.unique(text)
        # v = len(un)
        # voc = {}
        # for i in range(0, v):
        # voc[un[i]] = i
        #
        # return text, voc


def run():
    file = 'text_big.txt'
    alpha = 0.1
    epochs = 1000
    C = 1
    n = 10

    cbow = CBOW(file, alpha, epochs, C, n)


run()


