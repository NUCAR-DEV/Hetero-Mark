/*****************************************************************************/
/*IMPORTANT:  READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.         */
/*By downloading, copying, installing or using the software you agree        */
/*to this license.  If you do not agree to this license, do not download,    */
/*install, copy or use the software.                                         */
/*                                                                           */
/*                                                                           */
/*Copyright (c) 2005 Northwestern University                                 */
/*All rights reserved.                                                       */

/*Redistribution of the software in source and binary forms,                 */
/*with or without modification, is permitted provided that the               */
/*following conditions are met:                                              */
/*                                                                           */
/*1       Redistributions of source code must retain the above copyright     */
/*        notice, this list of conditions and the following disclaimer.      */
/*                                                                           */
/*2       Redistributions in binary form must reproduce the above copyright   */
/*        notice, this list of conditions and the following disclaimer in the */
/*        documentation and/or other materials provided with the distribution.*/
/*                                                                            */
/*3       Neither the name of Northwestern University nor the names of its    */
/*        contributors may be used to endorse or promote products derived     */
/*        from this software without specific prior written permission.       */
/*                                                                            */
/*THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS    */
/*IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED      */
/*TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY, NON-INFRINGEMENT AND         */
/*FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL          */
/*NORTHWESTERN UNIVERSITY OR ITS CONTRIBUTORS BE LIABLE FOR ANY DIRECT,       */
/*INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES          */
/*(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR          */
/*SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)          */
/*HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,         */
/*STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN    */
/*ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE             */
/*POSSIBILITY OF SUCH DAMAGE.                                                 */
/******************************************************************************/

/*
 * Mofified by: Leiming Yu (ylm@ece.neu.edu)
 * Mofified by: Yifan Sun (yifansun@coe.neu.edu)
 */

#ifndef SRC_HSA_KMEANS_HSA_KMEANS_BENCHMARK_H_
#define SRC_HSA_KMEANS_HSA_KMEANS_BENCHMARK_H_

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <fcntl.h>
#include <omp.h>
#include <unistd.h>
#include <float.h>
#include <memory.h>
#include <pthread.h>
#include <string>

#include "src/common/benchmark/benchmark.h"
#include "src/hsa/kmeans_hsa/kernels.h"

#ifndef FLT_MAX
#define FLT_MAX 3.40282347e+38
#endif

#define _CRT_SECURE_NO_DEPRECATE 1
#define RANDOM_MAX 2147483647

#define BLOCK_SIZE 64

class KmeansBenchmark : public Benchmark {
 public:
  KmeansBenchmark();
  ~KmeansBenchmark();

  void Initialize() override;
  void Run() override;
  void Verify() override;
  void Cleanup() override;
  void Summarize() override;

  void SetInputFileName(const char *filename) { filename_ = filename; }
  void SetMaxClusters(int max_clusters) { max_clusters_ = max_clusters; }
  void SetMinClusters(int min_clusters) { min_clusters_ = min_clusters; }

 private:
  std::string filename_;
  int num_points_;
  int num_features_;
  int max_clusters_;
  int min_clusters_;

  float **feature_;
  float *feature_transpose_;
  int *membership_;
  float **clusters_;
  float delta;

  float min_rmse_;
  int best_num_clusters_;

  //-----------------------------------------------------------------------//
  // I/O function
  //-----------------------------------------------------------------------//
  void Read();

  //-----------------------------------------------------------------------//
  // Cluster function
  //-----------------------------------------------------------------------//
  void Clustering();
  void CreateTemporaryMemory();
  void TransposeFeatures();
  void KmeansClustering(int num_clusters);
  void InitializeClusters(int num_clusters);
  void InitializeMembership();
  void UpdateMembership(int num_clusters);
  void UpdateClusterCentroids(int num_clusters);
  void DumpClusterCentroids(int num_clusters);
  void DumpMembership();
  float CalculateRMSE();

  //-----------------------------------------------------------------------//
  // Command line ouput
  //-----------------------------------------------------------------------//
  void DisplayResults();

  //-----------------------------------------------------------------------//
  // Clean functions
  //-----------------------------------------------------------------------//
  void Free_mem();

  int GetNumberPointsFromInputFile(FILE *input_file);
  int GetNumberFeaturesFromInputFile(FILE *input_file);
  void LoadFeaturesFromInputFile(FILE *input_file);
};

#endif  // SRC_HSA_KMEANS_HSA_KMEANS_BENCHMARK_H_
