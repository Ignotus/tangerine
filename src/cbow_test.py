from cbow import CBOW
from utils import  tokenize_files
from cbow_utils import tokenize_file, constructVocabulary, writeVocabulary,read_vocabulary
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import itertools
import profile

# train_dir = '../data/cbow/train/small/'
# output = '../data/cbow/vocab/small.txt'
# vocab_file = '../data/cbow/vocab/small.txt'


train_dir = '../data/training/'
vocab_file = '../data/vocabulary/20k.txt'

alpha = 0.25

C = 2
n = 5

EPOCHS = 5
MAX_VOCAB_SIZE = 20000
MAX_SENTENCES = 1000
MAX_LL_SENTENCES=1000

voc = constructVocabulary(train_dir)
writeVocabulary(voc, vocab_file)

print("Reading vocabulary " + vocab_file + "...")
index_to_word,word_to_index = read_vocabulary(vocab_file, MAX_VOCAB_SIZE)
print("Reading sentences and training CBOW ...")

start = timer()
myCbow = CBOW(C, n, index_to_word)
ERROR = []
EP = []
j = 0

def run():
        num_words = 0
        for i in range(EPOCHS):
            sentences1 = tokenize_files(word_to_index, train_dir)
            sentences2 = tokenize_files(word_to_index, train_dir) # for LL

            ERROR.append(myCbow.computeLL(itertools.islice(sentences1,MAX_LL_SENTENCES)))
            EP.append(i)

            for sentence in itertools.islice(sentences2, MAX_SENTENCES):
                myCbow.train(alpha, sentence)
                num_words += len(sentence)
            print("EPOCH " + str(i + 1) + "/" + str(EPOCHS) + " finished (" + str(num_words) + " words)")
            num_words = 0

#profile.run('run(); print')
run()
ERROR.append(myCbow.computeLL(itertools.islice(tokenize_files(word_to_index, train_dir),MAX_LL_SENTENCES)))
EP.append(EPOCHS)

print("- Took %.2f sec" % (timer() - start))
plt.plot(EP,ERROR)
plt.show()



