#!/bin/sh
cp bin/x86_64/Release/kmeans12 .
./kmeans12 -o -i ../data/kmeans/inpuGen/100000_34.txt 
rm kmeans12
