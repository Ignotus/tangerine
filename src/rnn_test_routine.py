#!/usr/bin/env python3

import sys
import itertools

from rnn import RNN
from rnn_extended import RNNExtended
from rnn_hierarchical_softmax import RNNHSoftmax

from utils import read_vocabulary, tokenize_files, index2word_to_VocabItems
from timeit import default_timer as timer

import pickle
import os.path

# Use preprocessed data from:
# http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz
# Contains one sentence tokenized per newline


MIN_WORD_COUNT=15
MAX_SENTENCES = 8000000000 # use all
MAX_LIKELIHOOD_SENTENCES = 5000

def write_vectors(words, rnn, filename):
    with open(filename, 'w') as output_file:
        for i, word in enumerate(words):
            vec = rnn.word_representation_inner(i)
            output_file.write(word[0] + " " + " ".join(str(f) for f in vec) + "\n")

def write_outer_vectors(words, rnn, filename):
    with open(filename, 'w') as output_file:
        for i, word in enumerate(words):
            vec = rnn.word_representation_outer(i)
            output_file.write(word[0] + " " + " ".join(str(f) for f in vec) + "\n")

def debug_sentence(words, sentence):
    print(" ".join(words[index][0] for index in sentence))

def testRNN(args, vocabulary_file, training_dir, testing_dir):
    print("Reading vocabulary " + vocabulary_file + "...")
    words, dictionary = read_vocabulary(vocabulary_file, min_count=MIN_WORD_COUNT)
    print("Vocabulary size: " + str(len(words)) + ", min-count=" + str(MIN_WORD_COUNT))

    print("Reading sentences and training RNN...")
    start = timer()

    if args.model == 'RNN':
        rnn = RNN(len(words), args.nhidden, use_relu=args.relu)
        if (args.load_weights and args.export_outer_weights):
            rnn.load(args.load_weights)
            write_outer_vectors(words, rnn, args.export_outer_weights)
            sys.exit(0)
    elif args.model == 'RNNExtended':
        rnn = RNNExtended(len(words), args.nhidden, args.class_size, use_relu=args.relu)
    elif args.model == 'RNNHSoftmax':
        vocItems = index2word_to_VocabItems(words)
        rnn = RNNHSoftmax(args.nhidden, vocItems, use_relu=args.relu)

    
    num_words = 0
    a = not os.path.isfile('testing_sentences.dump')
    if a:
        testing_sentences = tokenize_files(dictionary, testing_dir, subsample_frequent=False)
        lik_sentences = [sentence for sentence in itertools.islice(testing_sentences, MAX_LIKELIHOOD_SENTENCES)]
        pickle.dump(lik_sentences, open('testing_sentences.dump', 'wb'))
    else:
        lik_sentences = pickle.load(open('testing_sentences.dump', 'rb'))

    if a:
        for i in range(args.iter):
            print('Dumping sentences for the epoch %d' % (i))
            sentences = tokenize_files(dictionary, training_dir, subsample_frequent=False)
            pickle.dump([sentence for sentence in itertools.islice(sentences, MAX_SENTENCES)],
                        open('sentences_%d.dump' % (i), 'wb'))
        sys.exit()

    lr = args.learning_rate

    log_ll = rnn.log_likelihood(lik_sentences)
    print("Log-likelihood: %.2f" % (log_ll))
    for i in range(args.iter):
        epoch_start = timer()
        sentences = pickle.load(open('sentences_%d.dump' % (i), 'rb'))
        for idx, sentence in enumerate(sentences):
            rnn.train(sentence, lr=lr)
            num_words += len(sentence)
            if idx % 5000 == 0:
                print('%8d sentences processed. %d secs\r' % (idx, timer() - epoch_start), end='')
            # Checks a log-likelihood more frequent if use ReLU
            if args.relu and idx % 10000 == 0:
                new_log_ll = rnn.log_likelihood(lik_sentences)
                print("- Log-likelihood: %0.2f" % (new_log_ll))
                if new_log_ll < log_ll:
                    print('Log-likelihood has increased. Decreasing the learning rate..')
                    lr /= 2.0
                log_ll = new_log_ll

        print("Iteration " + str(i + 1) + "/" + str(args.iter) + " lr = %.8f" % (lr) + " finished (" + str(num_words) + " words)")
        new_log_ll = rnn.log_likelihood(lik_sentences)
        print("Log-likelihood: %.2f" % (new_log_ll))
        if new_log_ll < log_ll:
            print('Log-likelihood has increased. Decreasing the learning rate..')
            lr /= 2.0
        log_ll = new_log_ll
        print("- The Epoch Took %.2f sec" % (timer() - epoch_start))

        if args.export_file:
            print("- Writing vectors to file " + args.export_file  + "_" + str(i) + "...")
            write_vectors(words, rnn, args.export_file + "_" + str(i))

        num_words = 0


    print("- Took %.2f sec" % (timer() - start))

    if args.export_weights:
        rnn.export(args.export_weights)