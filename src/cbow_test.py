from cbow import CBOW
from utils import  tokenize_files
from cbow_utils import tokenize_file, constructVocabulary, writeVocabulary, tokenize_file, read_vocabulary
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import itertools
import profile

# train_dir = '../data/cbow/train/small/'
# output = '../data/cbow/vocab/small.txt'
# vocab_file = '../data/cbow/vocab/small.txt'



train_dir = '../data/training/small_1M'
#train_file = '../data/training/small_1M/news.txt'
vocab_file = '../data/vocabulary/news.txt'

# train_dir = '../data/training/medium_10M/'
# vocab_file = '../data/vocabulary/news.txt'

# might want to comment out when the voc. is already created
#voc = constructVocabulary(train_dir)
#writeVocabulary(voc, vocab_file,' ')


alpha = None
C = 5 # window size
n = 150 # the number of components in the hidden layer

EPOCHS = 1

MAX_VOCAB_SIZE = 5000 # use all
MAX_SENTENCES = 5000 # use all
MAX_LL_SENTENCES=5000


print("Reading vocabulary " + vocab_file + "...")
index_to_word,word_to_index = read_vocabulary(vocab_file, MAX_VOCAB_SIZE)
print("Reading sentences and training CBOW ...")

start = timer()
myCbow = CBOW(C, n, alpha,index_to_word)

# for performance plots
ERROR = []
EP = []

def tuneLR():
    return None;








def run():
        num_words = 0
        for i in range(0,EPOCHS):
            sentences1 = tokenize_files(word_to_index, train_dir)
            sentences2 = tokenize_files(word_to_index, train_dir) # for LL

            ERROR.append(myCbow.computeLL(itertools.islice(sentences1,MAX_LL_SENTENCES)))
            EP.append(i)

            for sentence in itertools.islice(sentences2, MAX_SENTENCES):
                myCbow.train(sentence)
                num_words += len(sentence)
            print("EPOCH " + str(i + 1) + "/" + str(EPOCHS) + " finished (" + str(num_words) + " words)")
            num_words = 0




#profile.run('run(); print')
run()
ERROR.append(myCbow.computeLL(itertools.islice(tokenize_files(word_to_index, train_dir),MAX_LL_SENTENCES)))
EP.append(EPOCHS)


print(EP)
print(ERROR)

print("- Took %.2f sec" % (timer() - start))
plt.plot(EP,ERROR)
plt.show()



