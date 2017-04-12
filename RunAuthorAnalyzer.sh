#!/usr/bin/env bash

DIR=$(dirname "$0")
PYTHONPATH="$DIR:$DIR/AST:$DIR/Utils:$DIR/Dataset"
export PYTHONPATH

declare -a arr=("Dataset/TestRepos/aws-sdk-java"
             "Dataset/TestRepos/camel"
             "Dataset/TestRepos/consulo")

for i in "${arr[@]}"
do
    python ${DIR}/AST/Analyze.py ${i}
done
