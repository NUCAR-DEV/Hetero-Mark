#!/bin/bash 

rm -f op.txt
for i in {1..100}
do
    ./bin/x86_64/Release/testSVM >> op.txt
done