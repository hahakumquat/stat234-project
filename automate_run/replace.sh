#!/bin/bash

params=(_game_ _model_ _ntrains_ _target_ _lr_ _batch_ _anneal_ _loss_ _reg_)

cat python.template > tmp.txt

for i in ${!params[@]}
do
    ibash=$((i+1))
    sed -i "s/${params[$i]}/${!ibash}/" tmp.txt
done
