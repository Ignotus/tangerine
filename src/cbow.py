import numpy as np
from cbow_utils import create_context_windows
from ada_grad import LR
from h_softmax import encode_huffman, hsm
from cbow_support import sigmoid


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

    def __init__(self, C, dim, alpha, vocab):
        self.vocab = vocab
        self.v = len(self.vocab)
        encode_huffman(self.vocab)  # encode the voc. as the huffman binary tree
        self.dim = dim
        self.C = C
        self.V = np.ones((self.v, self.dim))
        self.W = np.ones((self.dim, self.v))
        # self.alpha=alpha # used by pure SGD
        self.iAlpha = LR(alpha, self.v, dim)
        self.oAlpha = LR(alpha, self.v, dim)


    def train(self, sent):
        CWs = create_context_windows(sent, self.C)

        for cw in CWs:
            t = cw[0]  # the actual center word
            EH = np.zeros(self.dim)
            h = (1 / self.C) * np.sum(self.V[cw[1],], axis=0)

            classifiers = zip(self.vocab[t].path, self.vocab[t].code)
            for step, code in classifiers:
                p = sigmoid(self.W[:, step].T.dot(h))
                g = p - code
                EH += g * self.W[:, step]  # will be used to update input->hidden layer
                der = g * h
                self.W[:, step] -= self.oAlpha.getLR(step) * der  # 1. updating hidden -> output layer
                # self.W[:, target] -= self.alpha * der  # 1. updating hidden -> output layer
                self.oAlpha.updateTotalGrad(step, der)

            for w in cw[1]:
                # self.V[w, :] -= (self.alpha / self.C) * EH  # 2. updating input->hidden layer
                self.V[w, :] -= (self.iAlpha.getLR(w) / self.C) * EH  # 2. updating input->hidden layer
                self.iAlpha.updateTotalGrad(w, EH)


    def computeLL(self, sentences):
        LL = 0
        # iterate over sentences
        for sent in sentences:
            CWs = create_context_windows(sent, self.C)
            for cw in CWs:
                t = cw[0]  # the actual center word
                h = (1 / self.C) * np.sum(self.V[cw[1],], axis=0)
                LL += np.log(hsm(self.vocab[t], h, self.W))
        return LL






        # OUTDATED:


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
        #         r = self.y(j, h) - 1
        #     else:
        #         r = self.y(j, h)
        #     return r

    # # normal softmax
    # def y(self, j, h):
    # if (self.Cache_Z == None):
    # Z = 0
    # for i in range(0, self.v):
    # Z += np.exp(self.W[:, i].T.dot(h))
    #         if (Z == 0): return 0
    #         self.Cache_Z = Z
    #     return np.exp(self.W[:, j].T.dot(h)) / self.Cache_Z


    # hierarchical softmax
    # def hsm(self, j, h):
    #     classifiers = zip(self.vocab[j].path, self.vocab[j].code)
    #     res = 0
    #     for target, code in classifiers:
    #         t = 1 if code == 1 else -1
    #         res += sigmoid(t * self.W[:, target].T.dot(h))
    #     return res

    # def encode_huffman(self):
    #     # Build a Huffman tree
    #     vocab_size = self.v
    #     count = [t.count for t in self.vocab] + [1e15] * (vocab_size - 1)
    #     parent = [0] * (2 * vocab_size - 2)
    #     binary = [0] * (2 * vocab_size - 2)
    #
    #     pos1 = vocab_size - 1
    #     pos2 = vocab_size
    #
    #     for i in range(vocab_size - 1):
    #         # Find min1
    #         if pos1 >= 0:
    #             if count[pos1] < count[pos2]:
    #                 min1 = pos1
    #                 pos1 -= 1
    #             else:
    #                 min1 = pos2
    #                 pos2 += 1
    #         else:
    #             min1 = pos2
    #             pos2 += 1
    #
    #         # Find min2
    #         if pos1 >= 0:
    #             if count[pos1] < count[pos2]:
    #                 min2 = pos1
    #                 pos1 -= 1
    #             else:
    #                 min2 = pos2
    #                 pos2 += 1
    #         else:
    #             min2 = pos2
    #             pos2 += 1
    #
    #         count[vocab_size + i] = count[min1] + count[min2]
    #         parent[min1] = vocab_size + i
    #         parent[min2] = vocab_size + i
    #         binary[min2] = 1
    #
    #     # Assign binary code and path pointers to each vocab word
    #     root_idx = 2 * vocab_size - 2
    #     for i, token in enumerate(self.vocab):
    #         path = []  # List of indices from the leaf to the root
    #         code = []  # Binary Huffman encoding from the leaf to the root
    #
    #         node_idx = i
    #         while node_idx < root_idx:
    #             if node_idx >= vocab_size: path.append(node_idx)
    #             code.append(binary[node_idx])
    #             node_idx = parent[node_idx]
    #         path.append(root_idx)
    #
    #         # These are path and code from the root to the leaf
    #         token.path = [j - vocab_size for j in path[::-1]]
    #         token.code = code[::-1]



