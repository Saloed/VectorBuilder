#!/usr/bin/env bash

DIR=$(dirname "$0")
PYTHONPATH="$DIR:$DIR/AST:$DIR/Utils:$DIR/Dataset"
export PYTHONPATH

declare -a arr=("Dataset/TestRepos/aws-sdk-java"
             "Dataset/TestRepos/camel"
             "Dataset/TestRepos/cloudstack"
             "Dataset/TestRepos/consulo"
             "Dataset/TestRepos/eclipselink.runtime"
             "Dataset/TestRepos/FinanceAnalytics"
             "Dataset/TestRepos/MesquiteArchive"
             "Dataset/TestRepos/ppwcode")

for i in "${arr[@]}"
do
    python $DIR/AST/Analyze.py ${i}
done
