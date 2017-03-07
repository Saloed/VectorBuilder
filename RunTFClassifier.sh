#!/usr/bin/env bash

DIR=$(dirname "$0")
PYTHONPATH="$DIR:$DIR/AST:$DIR/Utils:$DIR/NN:$DIR/Dataset:$DIR/AuthorClassifier:$DIR/TFAuthorClassifier"
export PYTHONPATH
python "$DIR/TFAuthorClassifier/Creator.py"