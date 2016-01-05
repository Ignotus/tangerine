from cbow import CBOW, CBOWSMOpt,CBOWLROpt
from utils import tokenize_files, read_vocabulary, index2word_to_VocabItems, files_len
from cbow_utils import tokenize_file, constructVocabulary, writeVocabulary, tokenize_file, writeWordVectors
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from pylab import *
import numpy as np
import itertools
import profile



###### DATASETS ######
train_dir = '../data/training/medium_10M'
vocab_file = '../data/vocabulary/vocab_10M.txt'

#train_dir = '../data/training/small_1M'
#vocab_file = '../data/vocabulary/vocab_1M.txt'

#train_dir = '../data/hyperparameters/test/'
#vocab_file = '../data/hyperparameters/vocab.txt'


#tuning_train_dir = '../data/cbow/small/'
tuning_train_dir = '../data/hyperparameters/training/'
tuning_test_dir = '../data/hyperparameters/test/'
# tuning_all_dir= '../data/hyperparameters/all/'
tuning_vocab_file = '../data/hyperparameters/vocab.txt'

word_vectors_file='../word_vectors/cbow_100d_10C_delta_ns_10M_vectors.txt'



###### PARAMETERS ######
C = 10  # window size
n = 100  # the number of components in the hidden layer

file_lines=files_len(train_dir)

ALPHA=0.09 # the learning rate. !Set it to None to run the parameters tuning

EPOCHS = 1
MAX_VOCAB_SIZE = 500000000000 # use all
MAX_SENTENCES = 5000000000000  # use all
MAX_LL_SENTENCES = int(np.round(file_lines*1)) # use 1% out of all lines
LL_EVERY_SENTENCES = int(np.round(file_lines/5))
DEBUG=0 # change to 0 to switch off, to 1 to switch on print messages
SM_OPTIMIZATION= CBOWSMOpt.negative_sampling
LR_OPTIMIZATION= CBOWLROpt.ada_delta


# AdaDelta
ADELTA_NOISE      = 1e-6
ADELTA_DECAY_RATE = 0.95

###### CREATION OF A VOCAB. FILE (OPTIONAL) ######
#voc = constructVocabulary(tuning_all_dir)
#writeVocabulary(voc, tuning_vocab_file,' ')



def tuneLR():

    LLs=[]

    bestLR = 0
    bestLL = -9999999999999

    index_to_word, word_to_index = read_vocabulary(tuning_vocab_file, min_count=5)

    alphas = np.arange(0.01, 1, 0.05)
    print('-----------TUNING HYPERPARAMETERS---------')
    for alpha in alphas:
        myCbow = CBOW(index_to_word,C, n, alpha,SM_OPTIMIZATION,LR_OPTIMIZATION)
        sentences1 = tokenize_files(word_to_index, tuning_train_dir)
        sentences2 = tokenize_files(word_to_index, tuning_test_dir)
        for sentence in sentences1:
            myCbow.train(sentence)
        curLL = myCbow.computeLL(sentences2)
        LLs.append(curLL)
        if(DEBUG==1): print('current LR: '+str(alpha)+' LL: '+str(curLL))
        if (curLL > bestLL):
            bestLL = curLL
            bestLR = alpha


    print('best learning rate is: ' + str(bestLR) + ' with LL: ' + str(bestLL))
    print(LLs)
    print(alphas)
    if(DEBUG==1):
        width=0.35
        ind = np.arange(len(alphas))
        fig = plt.figure()
        plt.bar(ind, LLs,width,
                 alpha=0.9,
                 color='b',
                 label='Learning rates')
        plt.xlabel('learning rates')
        plt.ylabel('Log-likelihood')
        plt.xticks(ind+width/2,alphas)
        ax = gca()
        ax.margins(0.05, None)
        plt.savefig("../plots/cbow/LR.pdf")
        #plt.show()
    return alpha


def run():

    print("Reading vocabulary " + vocab_file + "...")
    index_to_word, word_to_index = read_vocabulary(vocab_file, min_count=5)
    # for performance plots
    ERROR = []
    EP = []

    start = timer()
    alpha = ALPHA if ALPHA else tuneLR()
    if ALPHA==None: print("- Took %.2f sec to tune hyper-params" % (timer() - start))

    num_samples=0
    if(SM_OPTIMIZATION==CBOWSMOpt.negative_sampling):
        num_samples= np.round(math.log(len(index_to_word),2))
        print("number of samples: "+str(num_samples))
    start = timer()
    myCbow = CBOW(index_to_word, C, n, alpha,SM_OPTIMIZATION,LR_OPTIMIZATION,num_negative_samples=num_samples,ada_delta_noise=ADELTA_NOISE,ada_delta_decay_rate=ADELTA_DECAY_RATE)
    num_words = 0

    ######## TRAINING ########
    print('----------STARTING TRAINING -----------')
    for i in range(0, EPOCHS):
        l=0 # lines processed
        j=0 ## LL runs

        sentences2 = tokenize_files(word_to_index, train_dir, subsample_frequent=False)  # for LL
        for sentence in itertools.islice(sentences2, MAX_SENTENCES):
            l+=1
            if((l%LL_EVERY_SENTENCES)==0):   print('Processed: ' +str(num_words))
            if(DEBUG==1 and ((l%LL_EVERY_SENTENCES)==0 )):
                j+=1
                sentences1 = tokenize_files(word_to_index, train_dir, subsample_frequent=False)
                LL=myCbow.computeLL(itertools.islice(sentences1, MAX_LL_SENTENCES))
                print('current LL is: '+str(LL))
                ERROR.append(LL)
                EP.append(j)
            myCbow.train(sentence)
            num_words += len(sentence)
        print("EPOCH " + str(i + 1) + "/" + str(EPOCHS) + " finished (" + str(num_words) + " words)")
        num_words = 0

    print("- Took %.2f sec to train the model" % (timer() - start))



    if(DEBUG==1):
        ERROR.append(myCbow.computeLL(itertools.islice(tokenize_files(word_to_index, train_dir), MAX_LL_SENTENCES)))
        EP.append(j+1)
        print(EP)
        print(ERROR)
        plt.plot(EP, ERROR)
        plt.show()
    else:
     ######## WORD VECTORS EXTRACTION ########
        print("writing vectors to "+word_vectors_file)
        wordVecs=[]
        for i, iw in enumerate(index_to_word):
            wordVecs.append([iw[0],myCbow.get_word_input_rep(i)])
        writeWordVectors(wordVecs,word_vectors_file)

#profile.run('run(); print')

#plot()
run()





