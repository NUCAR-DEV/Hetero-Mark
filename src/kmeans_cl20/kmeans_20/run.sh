#!/bin/sh
cp bin/x86_64/Release/kmeans .
./kmeans -o -i ../data/kmeans/kdd_cup
rm kmeans
