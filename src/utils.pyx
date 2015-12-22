#!/usr/bin/env python3
import numpy as np
import glob
import itertools

from spacy.parts_of_speech import ADJ, NO_TAG


from collections import defaultdict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import h_softmax

IGNORED_TOKEN = "IgnoreToken"
SUBSAMPLING_THRESHOLD = 10e-5

# Sub-sampling of frequent words: can improve both accuracy and speed for large data sets 
# Source: "Distributed Representations of Words and Phrases and their Compositionality"
def allow_with_prob(word, vocab_dict, total_wordcount):
    freq = float(vocab_dict[word][1]) / total_wordcount
    removal_prob = 1.0 - np.sqrt(SUBSAMPLING_THRESHOLD / freq)
    return np.random.random_sample() > removal_prob

def allow_word(word, vocab_dict, total_wordcount, subsample_frequent):
    allow = word.isalnum()
    if not allow:
        return False

    if not word in vocab_dict:
        return True # Will be replaced with IGNORED_TOKEN

    if subsample_frequent:
        allow = allow_with_prob(word, vocab_dict, total_wordcount)

    return allow

# Tokenizes the files in the given folder
# - Converts to lower case, removes punctuation
# - Yields sentences split up in words, represented by vocabulary index
# Files in data folder should be tokenized by sentence (one sentence per newline),
# Like in the 1B-words benchmark
def tokenize_files(vocab_dict, datafolder, subsample_frequent=False, nlp=None):
    total_dict_words = sum([value[1] for key, value in vocab_dict.items()])
    filenames = glob.glob(datafolder + "/*")
    for filename in filenames:
        with open(filename) as f:
            for sentence in f:
                # Use nltk tokenizer to split the sentence into words
                words = word_tokenize(sentence.lower())
                if nlp:
                    tokens = nlp(' '.join(words), tag=True, parse=False)
                    tags = [token.pos - ADJ + 1 for token in tokens]
                    # Filter (remove punctuation)
                    words = [(tag, word) for tag, word in zip(tags, words) if allow_word(word, vocab_dict, total_dict_words, subsample_frequent)]
                    # Replace words that are not in vocabulary with special token
                    words = [(tag, word) if word in vocab_dict else (NO_TAG, IGNORED_TOKEN) for tag, word in words]
                    # Yield the sentence as indices
                    if words:
                        yield [(tag, vocab_dict[word][0]) for tag, word in words]
                else:
                    # Filter (remove punctuation)
                    words = [word for word in words if allow_word(word, vocab_dict, total_dict_words, subsample_frequent)]
                    # Replace words that are not in vocabulary with special token
                    words = [word if word in vocab_dict else IGNORED_TOKEN for word in words]
                    # Yield the sentence as indices
                    if words:
                        yield [vocab_dict[word][0] for word in words]


def parse_word(string):
    parts = string.strip().split()
    return (parts[0], int(parts[1]))


# min_count: drop words from vocabulary that have count less than min_count
# max_size: limit the vocabulary size (use only the top 'max_size' words in the vocabulary list)
def read_vocabulary(filename, max_size=None, min_count=5):
    index_to_word = []
    with open(filename) as f:
        index_to_word = [parse_word(word) for word in itertools.islice(f, 0, max_size)]
        index_to_word = [word for word in index_to_word if word[1] >= min_count]

    # Mapping from "word" -> (index, occurrence_count)
    index_to_word.append([IGNORED_TOKEN, -1])
    word_to_index = dict([ (w[0], (i, w[1])) for i, w in enumerate(index_to_word)])
    return index_to_word, word_to_index

# Returns a list of VocabItem items with a prepared Hoffman binary tree
def index2word_to_VocabItems(index_to_word):
    vocItems = [h_softmax.VocabItem(word, count) for word, count in index_to_word]
    h_softmax.encode_huffman(vocItems)
    return vocItems

# Yield a list of lists of indices corresponding
# to context windows surrounding each word in the sentence
def create_context_windows(sentence, window_size):
    for i in range(0, len(sentence)):
        context_window = []

        for j in range(-(window_size), (window_size)+1):
            if j != 0 and i+j >= 0 and i+j < len(sentence):
                context_window.append(sentence[i+j])

        yield (sentence[i], context_window)

def files_len(folder):
    filenames = glob.glob(folder + "/*")
    for fname in filenames:
        with open(fname) as f:
            for i, l in enumerate(f):
                pass
    return i + 1