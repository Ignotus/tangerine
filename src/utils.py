#!/usr/bin/env python3
import numpy as np
import glob
from collections import defaultdict
from nltk.tokenize import word_tokenize
import itertools

IGNORED_TOKEN = "IGNORED_TOKEN"

# Tokenizes the files in the given folder
# - Converts to lower case, removes punctuation
# - Yields sentences split up in words, represented by vocabulary index
# Files in data folder should be tokenized by sentence (one sentence per newline),
# Like in the 1B-words benchmark

def tokenize_files(vocab_dict, datafolder):
    filenames = glob.glob(datafolder + "/*")
    for filename in filenames:
        with open(filename) as f:
            for sentence in f:
                # Use nltk tokenizer to split the sentence into words
                words = word_tokenize(sentence.lower())
                # Filter punctuation
                words = [word for word in words if word.isalpha()]
                # Replace words that are not in vocabulary with IGNORED token
                words = [word if word in vocab_dict else IGNORED_TOKEN for word in words]
                if words:
                    yield [vocab_dict[word] for word in words]

def read_vocabulary(filename, maxsize):
    index_to_word = []
    with open(filename) as f:
        index_to_word = [word.strip() for word in itertools.islice(f, 0, maxsize)]
    index_to_word.append(IGNORED_TOKEN)
    word_to_index = dict([ (w, i) for i, w in enumerate(index_to_word)])
    return index_to_word, word_to_index

# Yield a list of lists of indices corresponding
# to context windows surrounding each word in the sentence
def create_context_window(sentence, windowsize):
    # todo
    pass



