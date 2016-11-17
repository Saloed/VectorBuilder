#!/usr/bin/env bash

DIR=$(dirname "$0")
PYTHONPATH="$DIR:$DIR/AST:$DIR/Utils:$DIR/NN:$DIR/Dataset:$DIR/Embeddings"
export PYTHONPATH
python "$DIR/Embeddings/Train.py"