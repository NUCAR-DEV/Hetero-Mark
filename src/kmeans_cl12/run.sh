#!/bin/sh
cp bin/x86_64/Release/kmeans .
./kmeans -o -i ../data/kmeans/inpuGen/100000_34.txt 
rm kmeans
