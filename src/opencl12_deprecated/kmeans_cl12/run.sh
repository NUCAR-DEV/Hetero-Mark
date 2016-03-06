#!/bin/sh
cp bin/x86_64/Release/kmeans12 .
./kmeans12 -o -i ../../tool/100_34.txt
./kmeans12 -o -i ../../tool/1000_34.txt
./kmeans12 -o -i ../../tool/10000_34.txt
./kmeans12 -o -i ../../tool/100000_34.txt
rm kmeans12
