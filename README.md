# tangerine

[![License](http://img.shields.io/:license-mit-blue.svg)](http://doge.mit-license.org)

Natural Language Processing 1 course team project. It has been implemented based on the numpy and nltk libraries for the educational purpose and does not attempt to outperform models based on known libraries for numerical computation like Tensorflow or Theano. The project contains Python implementation of the following algorithms with their modifications:

- CBOW
- SkipGram
- RNN

## Dependencies

* nltk
* scipy
* numpy

### Running
```
python -u src/rnn_test.py --model RNNHSoftmax --relu --iter 10
```

For more informations
```
python -u src/rnn_test.py --help
```

## Copyright

Copyright (c) 2015 Minh Ngo, Arthur Bražinskas, Sander Lijbrink, Bryan Eikema

Copyright (c) 2016 Minh Ngo, Arthur Bražinskas

This project is distributed under the [MIT license](LICENSE). It's a part of the Natural Language Processing 1 course taught by Ivan Titov at the University of Amsterdam. Please follow the [UvA regulations governing Fraud and Plagiarism](http://student.uva.nl/en/az/content/plagiarism-and-fraud/plagiarism-and-fraud.html) in the case if you are a student.

## Known bugs

- SkipGram implementation doesn't work properly.
