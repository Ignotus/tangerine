# this file contains classes and functions that are used in hierarchical softmax
from cbow_support import sigmoid


# note that this function DOES NOT return anything
# instead it operated directly on the vocabulary objects
def encode_huffman(vocab):
    # Build a Huffman tree
    vocab_size = len(vocab)
    count = [t.count for t in vocab] + [1e15] * (vocab_size - 1)
    parent = [0] * (2 * vocab_size - 2)
    binary = [0] * (2 * vocab_size - 2)

    pos1 = vocab_size - 1
    pos2 = vocab_size

    for i in range(vocab_size - 1):
        # Find min1
        if pos1 >= 0:
            if count[pos1] < count[pos2]:
                min1 = pos1
                pos1 -= 1
            else:
                min1 = pos2
                pos2 += 1
        else:
            min1 = pos2
            pos2 += 1

        # Find min2
        if pos1 >= 0:
            if count[pos1] < count[pos2]:
                min2 = pos1
                pos1 -= 1
            else:
                min2 = pos2
                pos2 += 1
        else:
            min2 = pos2
            pos2 += 1

        count[vocab_size + i] = count[min1] + count[min2]
        parent[min1] = vocab_size + i
        parent[min2] = vocab_size + i
        binary[min2] = 1

    # Assign binary code and path pointers to each vocab word
    root_idx = 2 * vocab_size - 2
    for i, token in enumerate(vocab):
        path = []  # List of indices from the leaf to the root
        code = []  # Binary Huffman encoding from the leaf to the root

        node_idx = i
        while node_idx < root_idx:
            if node_idx >= vocab_size: path.append(node_idx)
            code.append(binary[node_idx])
            node_idx = parent[node_idx]
        path.append(root_idx)

        # These are path and code from the root to the leaf
        token.path = [j - vocab_size for j in path[::-1]]
        token.code = code[::-1]


### hierarchical softmax
# vi: the vocabulary item object corresponding to the word of interest (the one you compute probability for).
# It should have path and code attributes
# h: the computed values for the context of the word j.
# W: the matrix of output representations of inner nodes in the tree (used to be the output repr. matrix of words)

def hsm(vi, h, W):
    classifiers = zip(vi.path, vi.code)
    res = 0
    for step, code in classifiers:
        t = 1 if code == 1 else -1
        res += sigmoid(t * W[:, step].T.dot(h))
    return res


# the vocabulary item object that is initiated in cbow_utils.py read_vocabulary
class VocabItem:
    def __init__(self, word, count):
        self.word = word
        self.count = count
        self.path = None  # Path (list of indices) from the root to the word (leaf)
        self.code = None  # Huffman encoding


