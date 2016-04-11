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

#ifndef SRC_KMEANS_HSA_KMEANS_HSA_BENCHMARK_H_
#define SRC_KMEANS_HSA_KMEANS_HSA_BENCHMARK_H_

#include "src/kmeans/kmeans_benchmark.h"
#include "src/common/time_measurement/time_measurement.h"

#ifndef FLT_MAX
#define FLT_MAX 3.40282347e+38
#endif

#define _CRT_SECURE_NO_DEPRECATE 1
#define RANDOM_MAX 2147483647

#define BLOCK_SIZE 64

class KmeansHsaBenchmark : public KmeansBenchmark {
 private:
  float *feature_transpose_;
  int *membership_;
  float *clusters_;

  unsigned num_clusters_ = 0;
  float delta;
  float min_rmse_;
  int best_num_clusters_;

  void Clustering();
  void CreateTemporaryMemory();
  void FreeTemporaryMemory();
  void TransposeFeatures();
  void KmeansClustering(unsigned num_clusters);
  void InitializeClusters(unsigned num_clusters);
  void InitializeMembership();
  void UpdateMembership(unsigned num_clusters);
  void UpdateClusterCentroids(unsigned num_clusters);
  void DumpClusterCentroids(unsigned num_clusters);
  void DumpMembership();
  float CalculateRMSE();

 public:
  void Initialize() override;
  void Run() override;
  void Cleanup() override;
};

#endif  // SRC_KMEANS_HSA_KMEANS_HSA_BENCHMARK_H_