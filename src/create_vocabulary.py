import sys
import glob
import nltk

"""
Simple script to create an ordered vocabulary for a corpus
"""

# Reads text from a given data folder and counts word occurrences
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

# Writes vocabulary to a given file, sorted by word counts (most common on top)
def writeVocabulary(vocab, output_file):
    with open(output_file, 'w') as f:
        f.write("\n".join([word[0] for word in vocab.most_common()]))
    print("Vocabulary written to " + output_file)

if __name__ == '__main__':
    if len(sys.argv) > 2:
        vocab = constructVocabulary(sys.argv[1])
        writeVocabulary(vocab, sys.argv[2])
    else:
        print("Wrong arguments")
        print("- 1st argument should be input file directory")
        print("- 2nd argument should be output file")