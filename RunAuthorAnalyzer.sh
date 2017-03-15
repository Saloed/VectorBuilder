#!/usr/bin/env bash

DIR=$(dirname "$0")
PYTHONPATH="$DIR:$DIR/AST:$DIR/Utils:$DIR/Dataset"
export PYTHONPATH
python "$DIR/AST/Analyze.py"