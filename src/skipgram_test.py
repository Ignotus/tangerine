from skipgram import SkipGram, SkipGramOptimizations, SampleMode, LRMode
from utils import tokenize_files, read_vocabulary
from timeit import default_timer as timer

# Datasets and output
VOCAB_FILE      = "../data/skipgram/hyperparams_vocab.txt"
TRAINING_DIR    = "../data/skipgram/hyperparameters/training/"
TESTING_DIR     = "../data/skipgram/hyperparameters/test/"
OUTPUT_LOCATION = "./out"
OUTPUT_NAME     = "skipgram_vectors"

# External parameters
NUM_EPOCHS      = 1
MIN_OCCURRENCES = 5
SUBSAMPLE       = True

# Standard Skip-Gram parameters
OPTIMIZATION      = SkipGramOptimizations.negative_sampling
HIDDEN_LAYER_SIZE = 100
WINDOW_SIZE       = 4
LR_MODE           = LRMode.ada_delta

# Negative sampling parameters
NSAMPLE_MODE         = SampleMode.uniform
UNIGRAM_POWER        = 0.75
NUM_NEGATIVE_SAMPLES = 5

# AdaGrad / Vanilla SGD
LEARNING_RATE = 0.03

# AdaDelta
ADELTA_NOISE      = 1e-3
ADELTA_DECAY_RATE = 0.95

def test_skip_gram():
    print_parameters()

    # Read the vocabulary
    print("Reading vocabulary " + VOCAB_FILE + "...")
    words, dictionary = read_vocabulary(VOCAB_FILE, max_size=None, 
            min_count=MIN_OCCURRENCES)
    vocab_size = len(words)
    print("Read vocabulary, size: " + str(vocab_size))

    # Create the SkipGram model, start the timer
    start = timer()
    skip_gram = SkipGram(vocab_size, window_size=WINDOW_SIZE,
            hidden_layer_size=HIDDEN_LAYER_SIZE, optimization=OPTIMIZATION,
            vocab=words, num_negative_samples=NUM_NEGATIVE_SAMPLES,
            unigram_power=UNIGRAM_POWER, sample_mode=NSAMPLE_MODE,
            learning_rate=LEARNING_RATE, lr_mode=LR_MODE)

    LL = skip_gram.compute_LL(tokenize_files(dictionary, TESTING_DIR))
    print("The log-likelihood after no epoch is " + str(LL))

    # Do several training epochs over our training data
    for i in range(NUM_EPOCHS):
        start_epoch = timer()
        num_words = 0
        
        # Go over the entire training dataset
        for sentence in tokenize_files(dictionary, TRAINING_DIR, \
                subsample_frequent=SUBSAMPLE):
            skip_gram.train(sentence)
            num_words += len(sentence)

        # Print a status update
        print("Trained epoch #" + str(i + 1) + "/" + str(NUM_EPOCHS) + \
                ", processed " + str(num_words) + " words" + \
                ", took %.02f seconds." % (timer() - start_epoch))

        # Measure the log-likelihood
        LL = skip_gram.compute_LL(tokenize_files(dictionary, TESTING_DIR))
        print("The log-likelihood after this epoch is " + str(LL))

    # Report the performance results
    print("Training finished, took %.2f seconds" % (timer() - start))

    # Store the vector representations
    skip_gram.store_word_vectors(words, OUTPUT_LOCATION, OUTPUT_NAME)

def print_parameters():
    print("============================================================")
    print("Vocabulary file:    " + VOCAB_FILE)
    print("Training directory: " + TRAINING_DIR)
    print("Testing directory:  " + TESTING_DIR)
    print("Output directory:   " + OUTPUT_LOCATION)
    print("Output name prefix: " + OUTPUT_NAME + "\n")

    print("Number of epochs:         " + str(NUM_EPOCHS))
    print("Minimum word occurrences: " + str(MIN_OCCURRENCES))
    print("Subsampling enabled:      " + str(SUBSAMPLE) + "\n")

    print("Optimizations:              " + str(OPTIMIZATION.name))
    print("Hidden layer size:          " + str(HIDDEN_LAYER_SIZE))
    print("Window size:                " + str(WINDOW_SIZE))
    print("Learning rate mode:         " + str(LR_MODE.name) + "\n")

    print("Number of negative samples: " + str(NUM_NEGATIVE_SAMPLES))
    print("Negative sampling mode:     " + str(NSAMPLE_MODE.name))
    print("Unigram distribution power: " + str(UNIGRAM_POWER) + "\n")

    print("Learning rate: " + str(LEARNING_RATE) + "\n")

    print("AdaDelta noise:      " + str(ADELTA_NOISE))
    print("AdaDelta decay rate: " + str(ADELTA_DECAY_RATE))
    print("============================================================\n")

if __name__ == '__main__':
    test_skip_gram()
   