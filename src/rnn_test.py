#!/usr/bin/env python3

import argparse
from rnn_test_routine import *

if __name__ == '__main__':
    DESCRIPTION = """
        RNN Testing framework
        Copyright 2015 (c) Minh Ngo, Sander Lijbrink
    """
    parser = argparse.ArgumentParser(description=DESCRIPTION,
                                     formatter_class=argparse.RawTextHelpFormatter)

    rnn_mode = ['RNN', 'RNNExtended', 'RNNHSoftmax']
    parser.add_argument('--model', choices=rnn_mode, default='RNN', help='RNNLM Model mode')
    parser.add_argument('--iter', default=1, help='Number of iterations', type=int)
    parser.add_argument('--nhidden', default=100, help='Hidden layer size', type=int)
    parser.add_argument('--learning_rate', default=0.005, help='Learning rate', type=float)
    parser.add_argument('--data', default=1, help='Amount of data in millions', type=int)
    parser.add_argument('--relu', dest='relu', help='Enabling ReLU', action='store_true')
    parser.set_defaults(relu=False)
    parser.add_argument('--class_size', default=1000, help='Class size (is used only with RNNExtended models)', type=int)
    parser.add_argument('--export_file', default=None, help='File to which vectors are written', type=str)
    parser.add_argument('--export_weights', default=None, help='File to which RNN weights are exported', type=str)
    parser.add_argument('--load_weights', default=None, help='File from which to load RNN weights', type=str)
    parser.add_argument('--export_outer_weights', default=None, help='File to export outer representation (if it\'s supported)', type=str)
    args = parser.parse_args()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()

    #testRNN(args, "data/vocabulary/small.txt", "hyperparameters/training", "hyperparameters/training")
    testRNN(args, "data/vocabulary/vocab_%dM.txt" % (args.data),
            "data/%dM/training" % (args.data),
            "data/%dM/test/" % (args.data))

