#!/usr/bin/env python3
import numpy as np
import glob
import itertools

from collections import defaultdict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

IGNORED_TOKEN = "IgnoreToken"
SUBSAMPLING_THRESHOLD = 10e-5
STOPWORDS = set(stopwords.words('english'))

 # Sub-sampling of frequent words: can improve both accuracy and speed for large data sets 
# Source: "Distributed Representations of Words and Phrases and their Compositionality"
def remove_with_prob(word, vocab_dict, total_wordcount):
    freq = float(vocab_dict[word][1]) / total_wordcount
    removal_prob = 1.0 - np.sqrt(SUBSAMPLING_THRESHOLD / freq)
    return np.random.random_sample() < removal_prob

def allow_word(word, vocab_dict, total_wordcount, remove_stopwords=False, subsample_frequent=False, min_word_count=5):
    allow = True
    if remove_stopwords:
        allow = word.isalnum() and word not in STOPWORDS
    else:
        allow = word.isalnum()
    if not allow:
        return False

    if not word in vocab_dict:
        return True # Will be replaced with IGNORED_TOKEN

    if subsample_frequent:
        allow = remove_with_prob(word, vocab_dict, total_wordcount)
    if not allow:
        return False

    # Minimum word count
    if min_word_count > 1:
        allow = vocab_dict[word][1] >= min_word_count or word is IGNORED_TOKEN

    return allow

# Tokenizes the files in the given folder
# - Converts to lower case, removes punctuation
# - Yields sentences split up in words, represented by vocabulary index
# Files in data folder should be tokenized by sentence (one sentence per newline),
# Like in the 1B-words benchmark
def tokenize_files(vocab_dict, datafolder, remove_stopwords=False, subsample_frequent=False, min_word_count=5):
    total_dict_words = sum([value[1] for key, value in vocab_dict.items()])
    filenames = glob.glob(datafolder + "/*")
    for filename in filenames:
        with open(filename) as f:
            for sentence in f:
                # Use nltk tokenizer to split the sentence into words
                words = word_tokenize(sentence.lower())
                # Filter
                words = [word for word in words if allow_word(word, vocab_dict, total_dict_words, remove_stopwords, subsample_frequent, min_word_count)]
                # Replace words that are not in vocabulary with special token
                words = [word if word in vocab_dict else IGNORED_TOKEN for word in words]
                # Yield the sentence as indices
                if words:
                    yield [vocab_dict.get(word)[0] for word in words if vocab_dict.get(word)]

def parse_word(string):
    parts = string.strip().split()
    return (parts[0], int(parts[1]))

def read_vocabulary(filename, maxsize):
    index_to_word = []
    with open(filename) as f:
        index_to_word = [parse_word(word) for word in itertools.islice(f, 0, maxsize)]

    # Mapping from "word" -> (index, occurrence_count)
    index_to_word.append([IGNORED_TOKEN, -1])
    word_to_index = dict([ (w[0], (i, w[1])) for i, w in enumerate(index_to_word)])
    return index_to_word, word_to_index

# Yield a list of lists of indices corresponding
# to context windows surrounding each word in the sentence
def create_context_windows(sentence, window_size):
    for i in range(0, len(sentence)):
        context_window = []

        for j in range(-(window_size/2), (window_size/2)+1):
            if j != 0 and i+j >= 0 and i+j < len(sentence):
                context_window.append(sentence[i+j])

        yield (sentence[i], context_window)
