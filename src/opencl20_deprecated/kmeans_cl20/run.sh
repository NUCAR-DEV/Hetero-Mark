#!/bin/sh
cp bin/x86_64/Release/kmeans20 .
./kmeans20 -o -i ./data/kmeans/inpuGen/300_34.txt
rm kmeans20
