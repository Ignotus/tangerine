#!/usr/bin/env python3
import itertools
from skipgram import SkipGram
from utils import read_vocabulary, tokenize_files
from timeit import default_timer as timer

# Use preprocessed data from: 
# http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz
# Contains one sentence tokenized per newline

MAX_VOCAB_SIZE = 5000
MAX_SENTENCES = 1000
NUM_ITER = 5
HIDDEN_LAYER_SIZE = 20
WINDOW_SIZE = 2

def testSkipGram(vocabulary_file, training_dir):
    last_sentence = None
    print("Reading vocabulary " + vocabulary_file + "...")
    words, dictionary = read_vocabulary(vocabulary_file, MAX_VOCAB_SIZE)
    print("Reading sentences and training SkipGram...")
    start = timer()
    skip_gram = SkipGram(len(words), WINDOW_SIZE, HIDDEN_LAYER_SIZE)
    num_words = 0
    for i in range(NUM_ITER):
        sentences = tokenize_files(dictionary, training_dir)    
        for sentence in itertools.islice(sentences, MAX_SENTENCES):
            last_sentence = sentence
            skip_gram.train(sentence)
            num_words += len(sentence)

        ll = skip_gram.train(last_sentence, compute_ll=True)
        print("Iteration " + str(i + 1) + "/" + str(NUM_ITER) + " finished (" + str(num_words) + " words)")
        print("Log-likelihood: " + str(ll))

        num_words = 0

    print("- Took %.2f sec" % (timer() - start))

if __name__ == '__main__':
    testSkipGram("../data/vocabulary/20k.txt", "../data/training")
   