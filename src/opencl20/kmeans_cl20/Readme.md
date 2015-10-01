k-means clustering is a method of vector quantization, originally from
 signal processing, that is popular for cluster analysis in data mining.
 k-means clustering aims to partition n observations into k clusters in
 which each observation belongs to the cluster with the nearest mean,
 serving as a prototype of the cluster. In this implementation, we have
 varied the number of objects of 34 features and put them into 5 clusters.
 The input file contains features and attributes.

====== Hetero-Mark KMeans Benchmarks (OpenCL 1.2) ======
This benchmarks runs the KMeans Algorithm.

FileName[string]: -f --file (default = )
  File containing data to be clustered

Help[bool]: -h --help (default = false)
  Dump help information

Threshold[double]: -t --threshold (default = 0.001)
  Threshold value

binary[bool]: -b --binary (default = false)
  Input file is in binary format

cluster[bool]: -o --outputcluster (default = false)
  Output cluster center coordinates

max_nclusters[int]: -m --max (default = 5)
  Maximum number of clusters allowed

min_nclusters[int]: -n --min (default = 5)
  Minimum number of clusters allowed

nloops[int]: -l --loops (default = 1)
  Iteration for each number of clusters

rmse[bool]: -r --rmse (default = false)
  Calculate RMSE
