# tangerine

[![License](http://img.shields.io/:license-mit-blue.svg)](http://doge.mit-license.org)

Natural Language Processing 1 course team project

### Building
```
pushd src
  python setup.py build_ext --inplace
popd
```

### Running
```
python -u src/rnn_test.py --model RNNHSoftmax --relu --iter 10
```

For more informations
```
python -u src/rnn_test.py --help
```