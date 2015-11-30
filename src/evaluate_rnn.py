#!/usr/bin/env python3
import itertools
from rnn_extended import RNNExtended
from rnn_extended_relu import RNNExtendedReLU

from utils import read_vocabulary, tokenize_files
from timeit import default_timer as timer

MAX_VOCAB_SIZE = 10000000 ## use all
NUM_ITER = 1

def write_vectors(words, rnn, filename):
    with open(filename, 'w') as output_file:
        for i, word in enumerate(words):
            vec = rnn.word_representation(i)
            output_file.write(word[0] + " " + " ".join(str(f) for f in vec))

def evaluateRNN(vocabulary_file, training_dir, vector_file, hidden_layer_size):
    print("Reading vocabulary " + vocabulary_file + "...")
    words, dictionary = read_vocabulary(vocabulary_file, MAX_VOCAB_SIZE)
    print("Reading sentences and training RNN...")
    start = timer()
    rnn = RNNExtended(len(words), hidden_layer_size, class_size=1000)
    num_words = 0
    for i in range(NUM_ITER):
        print("Iteration: " + str(i + 1))
        sentences = tokenize_files(dictionary, training_dir, remove_stopwords=True)
        sent = 0
        for sentence in sentences:
            rnn.train(sentence)
            num_words += len(sentence)
            sent += 1
            print("Trained sentence " + str(sent))

        num_words = 0

    print("- Took %.2f sec" % (timer() - start))
    print("- Saving word representations to file " + vector_file)
    write_vectors(words, rnn, vector_file)

if __name__ == '__main__':
    evaluateRNN("../data/vocabulary/small.txt", "../data/training/small_1M", "word_vectors_1M_hidden50.txt", 50)
    #evaluateRNN("../data/vocabulary/small.txt", "../data/training/small_1M", "word_vectors_1M_hidden80.txt", 80)
