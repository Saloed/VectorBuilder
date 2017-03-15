#!/usr/bin/env bash

DIR=$(dirname "$0")
PYTHONPATH="$DIR:$DIR/AST:$DIR/Utils:$DIR/Dataset:$DIR/TFAuthorClassifier"
export PYTHONPATH
python "$DIR/TFAuthorClassifier/Train.py"