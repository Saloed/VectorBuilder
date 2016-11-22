#!/usr/bin/env bash

DIR=$(dirname "$0")
PYTHONPATH="$DIR:$DIR/AST:$DIR/Utils:$DIR/NN:$DIR/Dataset:$DIR/AuthorClassifier"
export PYTHONPATH
python "$DIR/NetTest/Train.py"