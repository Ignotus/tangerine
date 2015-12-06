import numpy as np
import glob
from collections import defaultdict
from nltk.tokenize import word_tokenize
import itertools
import nltk
from h_softmax import VocabItem

IGNORED_TOKEN = "IGNORED_TOKEN"

def tokenize_file(vocab_dict, file):
        with open(file) as f:
            for sentence in f:
                # Use nltk tokenizer to split the sentence into words
                words = word_tokenize(sentence.lower())
                # Filter punctuation
                words = [word for word in words if word.isalnum()]
                # Replace words that are not in vocabulary with IGNORED token
                words = [word if word in vocab_dict else IGNORED_TOKEN for word in words]
                if words:
                  yield [vocab_dict[word] for word in words]

def read_vocabulary(filename, maxsize,sep=' '):
    index_to_word = []
    with open(filename) as f:
        for word in itertools.islice(f, 0, maxsize):
            splt=word.split(sep)
            index_to_word.append(VocabItem(splt[0],int(splt[1].strip())))
    index_to_word.append(VocabItem(IGNORED_TOKEN,0))
    word_to_index = dict([(w.word, (i, w.count)) for i, w in enumerate(index_to_word)])
    return index_to_word, word_to_index


def create_context_windows(sentence, window_size):
    for i in range(0, len(sentence)):
        context_window = []

        for j in range(-(window_size), (window_size)+1):
            if j != 0 and i+j >= 0 and i+j < len(sentence):
                context_window.append(sentence[i+j])
        yield (sentence[i], context_window)

def constructVocabulary(folder):
    filenames = glob.glob(folder + "/*")
    freqs = nltk.FreqDist()
    print("Creating vocabulary...")

    for i, filename in enumerate(filenames):
        with open(filename) as f:
            print("- Processing file " + filename + "...")
            words = nltk.word_tokenize(f.read().lower())
            words = [word for word in words if word.isalnum()]
            freqs.update(words)
    return freqs

def writeVocabulary(vocab, output_file,sep=' '):
    with open(output_file, 'w') as f:
        f.write("\n".join([sep.join((word[0],str(word[1]))) for word in vocab.most_common()]))
    print("Vocabulary written to " + output_file)


def writeWordVectors(wordVecs, filename):
    with open(filename, 'w') as output_file:
        for i, wordVec in enumerate(wordVecs):
            output_file.write(wordVec[0] + " " + " ".join(str(f) for f in wordVec[1])+"\n")
