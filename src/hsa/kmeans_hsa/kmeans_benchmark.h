/*
 * Hetero-mark
 *
 * Copyright (c) 2015 Northeastern University
 * All rights reserved.
 *
 * Developed by:
 *   Northeastern University Computer Architecture Research (NUCAR) Group
 *   Northeastern University
 *   http://www.ece.neu.edu/groups/nucar/
 *
 * Modified by: Leiming Yu (ylm@ece.neu.edu)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a 
 * copy of this software and associated documentation files (the "Software"), 
 * to deal with the Software without restriction, including without limitation 
 * the rights to use, copy, modify, merge, publish, distribute, sublicense, 
 * and/or sell copies of the Software, and to permit persons to whom the 
 * Software is furnished to do so, subject to the following conditions:
 * 
 *   Redistributions of source code must retain the above copyright notice, 
 *   this list of conditions and the following disclaimers.
 *
 *   Redistributions in binary form must reproduce the above copyright 
 *   notice, this list of conditions and the following disclaimers in the 
 *   documentation and/or other materials provided with the distribution.
 *
 *   Neither the names of NUCAR, Northeastern University, nor the names of 
 *   its contributors may be used to endorse or promote products derived 
 *   from this Software without specific prior written permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
 * DEALINGS WITH THE SOFTWARE.
 */

#ifndef SRC_HSA_KMEANS_HSA_KMEANS_BENCHMARK_H_
#define SRC_HSA_KMEANS_HSA_KMEANS_BENCHMARK_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <fcntl.h>
#include <omp.h>
#include <unistd.h>
#include <float.h>
#include <memory.h>
#include <pthread.h>
#include <sys/time.h>

#include "src/common/benchmark/benchmark.h"
#include "src/hsa/kmeans_hsa/kernels.h"

#ifndef FLT_MAX
#define FLT_MAX 3.40282347e+38
#endif

#define _CRT_SECURE_NO_DEPRECATE 1
#define RANDOM_MAX 2147483647

#ifdef RD_WG_SIZE_0_0
#define BLOCK_SIZE RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
#define BLOCK_SIZE RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
#define BLOCK_SIZE RD_WG_SIZE
#else
#define BLOCK_SIZE 256
#endif

#ifdef RD_WG_SIZE_1_0
#define BLOCK_SIZE2 RD_WG_SIZE_1_0
#elif defined(RD_WG_SIZE_1)
#define BLOCK_SIZE2 RD_WG_SIZE_1
#elif defined(RD_WG_SIZE)
#define BLOCK_SIZE2 RD_WG_SIZE
#else
#define BLOCK_SIZE2 256
#endif

extern double wtime(void);

class KmeansBenchmark : public Benchmark {
 public:
  KmeansBenchmark();
  ~KmeansBenchmark();

  void Initialize() override;
  void Run() override;
  void Verify() override;
  void Cleanup() override;
  void Summarize() override;

 private:
  //-----------------------------------------------------------------------//
  // Host Parameters
  //-----------------------------------------------------------------------//

  // command line options
  char *filename;
  int isBinaryFile;
  int isOutput;
  int npoints;
  int nfeatures;
  int max_nclusters;
  int min_nclusters;
  int isRMSE;
  int nloops; // number of iterations for each cluster
  int index;  // number of iteration to reach the best RMSE
  float threshold;

  float **feature; // host feature

  //-----------------  Clustering parameters ------------------------------//
  int nclusters;   // number of clusters
  int *membership; // which cluster a data point belongs to
  // hold coordinates of cluster centers
  float **tmp_cluster_centres; // pointer to the clusters
  float **cluster_centres;     // pointer to the clusters

  //-----------------  Create_mem -----------------------------------------//
  int *membership_OCL;

  //----------------- Kmeans_clustering parameters ------------------------//
  float **clusters;     // out: [nclusters][nfeatures]
  int *initial;         // used to hold the index of points not yet selected
  int *new_centers_len; // [nclusters]: no. of points in each cluster
  float **new_centers;  // [nclusters][nfeatures]
  float delta;          // if the point moved

  //----------------- rms_err parameters ----------------------------------//
  float rmse; // RMSE for each clustering
  float min_rmse;
  float min_rmse_ref;
  int best_nclusters;

  //-----------------------------------------------------------------------//
  // Device Parameters
  //-----------------------------------------------------------------------//
  // device memory
  float *d_feature; // device feature
  float *d_feature_swap;
  float *d_cluster; // cluster
  int *d_membership;

  //-----------------------------------------------------------------------//
  // I/O function
  //-----------------------------------------------------------------------//
  void Read();

  //-----------------------------------------------------------------------//
  // Cluster function
  //-----------------------------------------------------------------------//
  void Clustering();
  void Create_mem();
  void Swap_features();
  void Kmeans_clustering();
  void Kmeans_ocl();

  //-----------------------------------------------------------------------//
  // rms function
  //-----------------------------------------------------------------------//
  float euclid_dist_2(float *, float *);
  int find_nearest_point(float *, float **);
  void RMS_err();

  //-----------------------------------------------------------------------//
  // Command line ouput
  //-----------------------------------------------------------------------//
  void Display_results();

  //-----------------------------------------------------------------------//
  // Clean functions
  //-----------------------------------------------------------------------//
  void Free_mem();
};

#endif  // SRC_HSA_KMEANS_HSA_KMEANS_BENCHMARK_H_
