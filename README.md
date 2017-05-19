# Author identification using TB-CNN

Author identification approach based on tree-based convolution neural network ([TB-CNN](https://arxiv.org/abs/1409.5718)).

### Repository organization
 * [AST](AST) package contains methods to build a CST from a Java source code
 
 * [Word2Vec](Word2Vec) contains approach to learn CST token embeddings from a source code 
 
 * [AuthorClassifier](TFAuthorClassifier) contains our implementation of tree-based convolution
    and methods to prepare dataset, train and test network

### How to use

You need [Python 3.6+](https://www.python.org/downloads/release/python-361/) 
and [tensorflow 1.0.0](https://www.tensorflow.org/) to be installed.

Our implementation take authorship information from Git repository, so you need analyzing sources
to be under Git version control system. 
1. Create a project file using code from [AST](AST) package manually or
 using [special script](RunAuthorAnalyzer.sh)
2. Build a dataset file using [prepare script](TFAuthorClassifier/DataPreparation.py) and project file
3. Write path to your dataset file in [train script](TFAuthorClassifier/Train.py) and 
[test script](TFAuthorClassifier/Test.py)
4. Start training process manually or using [special script](RunTFClassifier.sh)
5. Write path to your trained model in a [train script](TFAuthorClassifier/Train.py) and
    check performance of network