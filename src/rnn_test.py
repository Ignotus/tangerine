#!/usr/bin/env python3

import sys
import itertools
import argparse

from rnn import RNN
from rnn_relu import RNNReLU
from rnn_extended import RNNExtended
from rnn_extended_relu import RNNExtendedReLU

from utils import read_vocabulary, tokenize_files
from timeit import default_timer as timer

# Use preprocessed data from:
# http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz
# Contains one sentence tokenized per newline

MAX_VOCAB_SIZE = 10000
MAX_SENTENCES = 100
MAX_LIKELIHOOD_SENTENCES = 100

def testRNN(args, vocabulary_file, training_dir):
    print("Reading vocabulary " + vocabulary_file + "...")
    words, dictionary = read_vocabulary(vocabulary_file, MAX_VOCAB_SIZE)
    print("Reading sentences and training RNN...")
    start = timer()

    if args.model == 'RNN':
        rnn = RNN(len(words), args.nhidden)
    elif args.model == 'RNNReLU':
        rnn = RNNReLU(len(words), args.nhidden)
        rnn.grad_threshold = args.maxgrad
    elif args.model == 'RNNExtended':
        rnn = RNNExtended(len(words), args.nhidden, args.class_size)
    else:
        rnn = RNNExtendedReLU(len(words), args.nhidden, args.class_size)
        rnn.grad_threshold = args.maxgrad
    
    num_words = 0
    sentences = tokenize_files(dictionary, training_dir)
    lik_sentences = [sentence for sentence in itertools.islice(sentences, MAX_LIKELIHOOD_SENTENCES)]
    for i in range(args.iter):
        sentences = tokenize_files(dictionary, training_dir, remove_stopwords=True)
        for sentence in itertools.islice(sentences, MAX_SENTENCES):
            # Todo, create context window for each sentence?
            rnn.train(sentence)
            num_words += len(sentence)

        print("Iteration " + str(i + 1) + "/" + str(args.iter) + " finished (" + str(num_words) + " words)")
        print("Log-likelihood: %.2f" % (rnn.log_likelihood(lik_sentences)))
        num_words = 0

    print("- Took %.2f sec" % (timer() - start))

if __name__ == '__main__':
    DESCRIPTION = """
        RNN Testing framework
        Copyright 2015 (c) Minh Ngo, Sander Lijbrink
    """
    parser = argparse.ArgumentParser(description=DESCRIPTION,
                                     formatter_class=argparse.RawTextHelpFormatter)

    rnn_mode = ['RNN', 'RNNReLU', 'RNNExtended', 'RNNExtendedReLU']
    parser.add_argument('--model', choices=rnn_mode, default='RNN', help='RNNLM Model mode')
    parser.add_argument('--iter', default=5, help='Number of iterations')
    parser.add_argument('--nhidden', default=20, help='Hidden layer size')
    parser.add_argument('--maxgrad', default=0.01, help='Gradient clipping threshold (is used only with ReLU models)')
    parser.add_argument('--class_size', default=10000, help='Class size (is used only with RNNExtended models)')

    args = parser.parse_args()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()

    testRNN(args, "../data/vocabulary/small.txt", "../data/training/small_1M")

