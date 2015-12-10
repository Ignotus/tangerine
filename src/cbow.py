import numpy as np
from cbow_utils import create_context_windows
from utils import index2word_to_VocabItems
import ada_grad
import ada_delta
from h_softmax import encode_huffman, hsm
from cbow_support import sigmoid
from enum import Enum
from unigram import UnigramDistribution


class CBOWSMOpt(Enum):
    none = 1
    negative_sampling = 2
    hierarchical_softmax = 3


class CBOWLROpt(Enum):
    none = 1  # pure SGD
    ada_grad = 2
    ada_delta = 3


class CBOW:
    V = None  # input matrix
    W = None  # output matrix
    vocab = None

    v = None  # the size of the vocab.
    dim = None  # the number of word vector components.

    C = None  # the size of the context window

    iAlpha = None  # the learning rate for input parameters
    oAlpha = None  # for output parameters
    alpha = None  # raw learning rate

    smOpt = None
    lrOpt = None

    unigram = None  # unigram distribution
    K = None  # number of negative samples

    DR = None  # adaDelta decay noise
    DN = None  # adaDelta noise

    cache_Z = None  # a cached constant that can use used to speed-up softmax


    def __init__(self, vocab, C=2, dim=100, alpha=0.15, smOpt=CBOWSMOpt.none, lrOpt=CBOWLROpt.none,
                 num_negative_samples=5, unigram_power=0.75, ada_delta_noise=1e-3, \
                 ada_delta_decay_rate=0.95):
        self.v = len(vocab)
        self.lrOpt = lrOpt
        self.smOpt = smOpt
        if (smOpt == CBOWSMOpt.hierarchical_softmax):
            self.vocab = index2word_to_VocabItems(vocab)
        if (smOpt == CBOWSMOpt.none):
            self.vocab = vocab
        if (smOpt == CBOWSMOpt.negative_sampling):
            self.unigram = UnigramDistribution(vocab, unigram_power)
            self.K = num_negative_samples
        self.alpha = alpha  # used by pure SGD
        self.dim = dim
        self.C = C

        # weights initialization
        self.V = np.random.uniform(low=-0.5 / dim, high=0.5 / dim, size=(self.v, dim))
        self.W = np.zeros(shape=(dim, self.v))


        # initialize learning params and objects
        if (lrOpt == CBOWLROpt.ada_grad):
            self.iAlpha = ada_grad.LR(alpha, self.v, dim)
            self.oAlpha = ada_grad.LR(alpha, self.v, dim)
        if (lrOpt == CBOWLROpt.ada_delta):
            self.iAlpha = ada_delta.LR(self.v, dim, noise=ada_delta_noise, \
                                       decay_rate=ada_delta_decay_rate)
            self.oAlpha = ada_delta.LR(self.v, dim, noise=ada_delta_noise, \
                                       decay_rate=ada_delta_decay_rate)


    def train(self, sent):
        if (self.smOpt == CBOWSMOpt.hierarchical_softmax):
            self.__train_hsm(sent)
        if (self.smOpt == CBOWSMOpt.negative_sampling):
            self.__train_ns(sent)
        if (self.smOpt == CBOWSMOpt.none):
            self.__train_plain(sent)

    def computeLL(self, sentences):
        LL = 0
        # iterate over sentences
        for sent in sentences:
            CWs = create_context_windows(sent, self.C)
            for cw in CWs:
                cwl = len(cw[1]) if (len(cw[1]) != 0) else 1  # the number of context words
                t = cw[0]  # the actual center word
                h = (1 / cwl) * np.sum(self.V[cw[1], :], axis=0)
                prob = self.___prob(t, h)
                LL += (0 if prob <= 0 else np.log(prob))
        return LL


    # THIS MIGHT NOT WORK FOR HIERARCHICAL SOFTMAX!!!
    def get_word_output_rep(self, word_idx):
        return self.W[:, word_idx]

    def get_word_input_rep(self, word_idx):
        return self.V[word_idx, :]


    def __train_ns(self, sent):
        CWs = create_context_windows(sent, self.C)

        for cw in CWs:
            t = cw[0]  # the actual center word
            cwl = len(cw[1]) if (
                len(cw[1]) != 0) else 1  # the number of context words, note that can't have division by 0
            h = (1 / cwl) * np.sum(self.V[cw[1], :], axis=0)
            samples = self.__draw_negative_samples(self.K)
            # add positive sample to negative samples
            # np.append(samples, t)  # this one might be linear, might want to find a way around to make it constant
            EH = np.zeros(self.dim)


            # TODO: need to make one loop and concat. samples into one array
            # for positive sample
            o_repr = self.W[:, t]
            p = sigmoid(o_repr.T.dot(h))
            p -= 1
            g= p * h
            EH += p * o_repr
            self.W[:, t] -= self.___get_and_update__LR(t, 2,g) * g  # 1. updating hidden->output layer

            # for negative sample
            for s in samples:
                o_repr = self.W[:, s]
                p = sigmoid(o_repr.T.dot(h))
                p -= 1 if s == t else 0  # -1 if positive sample 0 otherwise
                g= p * h
                EH += p * o_repr
                self.W[:, s] -= self.___get_and_update__LR(s, 2,g) * g  # 1. updating hidden->output layer

            for w in cw[1]:
                self.V[w, :] -= (self.___get_and_update__LR(w, 1,EH/cwl) / cwl) * EH  # 2. updating input->hidden layer


    def __train_plain(self, sent):
        CWs = create_context_windows(sent, self.C)

        for cw in CWs:
            t = cw[0]  # the actual center word
            cwl = len(cw[1]) if (
                len(cw[1]) != 0) else 1  # the number of context words, note that can't have division by 0
            h = (1 / cwl) * np.sum(self.V[cw[1], :], axis=0)
            EH = np.zeros(self.dim)
            self.cache_Z = None  # reset cached Z
            for j in range(0, self.v):
                p = self.sm(j, h)
                p -= 1 if j == t else 0
                EH += p * self.W[:, j]
                g = p * h
                self.W[:, j] -= self.___get_and_update__LR(j, 2,g) * g  # 1. updating hidden->output layer
            for w in cw[1]:
                self.V[w, :] -= (self.___get_and_update__LR(w, 1,EH/cwl) / cwl) * EH  # 2. updating input->hidden layer


    def __train_hsm(self, sent):
        CWs = create_context_windows(sent, self.C)

        for cw in CWs:
            t = cw[0]  # the actual center word
            EH = np.zeros(self.dim)
            cwl = len(cw[1]) if (
                len(cw[1]) != 0) else 1  # the number of context words, note that can't have division by 0
            h = (1 / cwl) * np.sum(self.V[cw[1], :], axis=0)

            classifiers = zip(self.vocab[t].path, self.vocab[t].code)
            for step, code in classifiers:
                p = sigmoid(self.W[:, step].T.dot(h))
                g = p - code
                EH += g * self.W[:, step]  # will be used to update input->hidden layer
                der = g * h
                self.W[:, step] -= self.___get_and_update__LR(step, 2,der) * der  # 1. updating hidden -> output layer
                # self.___update_LR(step, 2, der)  # updating parameters related to the LR of output layer weights

            for w in cw[1]:
                self.V[w, :] -= (self.___get_and_update__LR(w, 1,EH/cwl) / cwl) * EH  # 2. updating input->hidden layer
                # self.___update_LR(w, 1, EH / cwl)  # updating parameters related to the LR of input layer weights


    def __draw_negative_samples(self, amount):
        return self.unigram.sample(amount)



    def ___prob(self, t, h):
        if self.smOpt == CBOWSMOpt.negative_sampling:
            samples = self.__draw_negative_samples(self.K)
            p = sigmoid(self.W[:, t].T.dot(h))
            for s in samples:
                p *= sigmoid(-self.W[:, s].T.dot(h))
        if self.smOpt == CBOWSMOpt.hierarchical_softmax:
            p = hsm(self.vocab[t], h, self.W)
        if self.smOpt == CBOWSMOpt.none:
            p = self.sm(t, h)
        return p

    # returns different learning rates depending on the learning rate optimization
    # id: the id of a word or inner node in a huffman's tree
    # layer: 1: input layer, 2: output layer
    def ___get_and_update__LR(self, id=0, layer=1, der=None):
        if not (layer in [1, 2]): raise Exception("Invalid layer number")

        if (self.lrOpt == CBOWLROpt.ada_delta):
            if (layer == 1):
                lr=self.iAlpha.get_and_update_LR(id,der)
            else:
                lr= self.oAlpha.get_and_update_LR(id,der)
        if (self.lrOpt == CBOWLROpt.ada_grad):
            if (layer == 1):
                lr=self.iAlpha.getLR(id)
                self.iAlpha.updateTotalGrad(id, der)
            else:
                lr= self.oAlpha.getLR(id)
                self.oAlpha.updateTotalGrad(id, der)
        if (self.lrOpt == CBOWLROpt.none):
                lr= self.alpha
        return lr

    # normal softmax
    def sm(self, j, h):
        if (self.cache_Z == None):
            Z = 0
            for i in range(0, self.v):
                Z += np.exp(self.W[:, i].T.dot(h))
            if (Z == 0): return 0
            self.cache_Z = Z
        return np.exp(self.W[:, j].T.dot(h)) / self.cache_Z

    ## OUTDATED:


    # def getEH(self, t, h):
    # res = 0
    # for j in range(0, self.v):
    # res += self.getE(j, t, h) * (self.W[:, j])
    # return res
    #
    #
    # # h: hidden neuron values
    # def getE(self, j, t, h):
    # if (t == j):
    # r = self.y(j, h) - 1
    # else:
    # r = self.y(j, h)
    # return r



        #
        # def computeLL(self, sentences):
        # LL = 0
        # # iterate over sentences
        #     for sent in sentences:
        #         CWs = create_context_windows(sent, self.C)
        #         for cw in CWs:
        #             cwl = len(cw[1]) if (len(cw[1]) != 0) else 1  # the number of context words
        #             t = cw[0]  # the actual center word
        #             h = (1 / cwl) * np.sum(self.V[cw[1], :], axis=0)
        #             prob = hsm(self.vocab[t], h, self.W)
        #             LL += (0 if prob == 0 else np.log(prob))
        #     return LL