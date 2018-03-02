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

#ifndef SRC_KMEANS_KMEANS_BENCHMARK_H_
#define SRC_KMEANS_KMEANS_BENCHMARK_H_

#include <cfloat>
#include <cmath>
#include <string>
#include "src/common/benchmark/benchmark.h"
#include "src/common/time_measurement/time_measurement.h"

class KmeansBenchmark : public Benchmark {
 protected:
  const unsigned kBlockSize = 256;

  std::string filename_ = "";
  double threshold_ = 0.001;
  unsigned max_num_clusters_ = 5;
  unsigned min_num_clusters_ = 5;
  unsigned num_loops_ = 1;

  unsigned num_points_ = 0;
  unsigned num_features_ = 0;

  float *host_features_;
  float *feature_transpose_;
  int *membership_;
  float *clusters_;

  float cpu_min_rmse_;

  unsigned num_clusters_ = 0;
  float delta_;
  float min_rmse_;
  int best_num_clusters_;

  void DumpMembership();
  void DumpClusterCentroids(unsigned num_clusters);
  void DumpFeatures();
  float CalculateRMSE();
  void TransposeFeaturesCpu();
  void KmeansClusteringCpu(unsigned num_clusters);
  virtual void InitializeClusters(unsigned num_clusters);
  virtual void InitializeMembership();
  void UpdateMembershipCpu(unsigned num_clusters);
  void UpdateClusterCentroids(unsigned num_clusters);

 public:
  KmeansBenchmark() : Benchmark() {}
  void Initialize() override;
  void Run() override {}
  void Verify() override;
  void Summarize() override;
  void Cleanup() override;

  // Setters
  void setFilename(std::string filename) { filename_ = filename; }
  void setThreshold(double threshold) { threshold_ = threshold; }
  void setMaxNumClusters(unsigned max_num_clusters) {
    max_num_clusters_ = max_num_clusters;
  }
  void setMinNumClusters(unsigned min_num_clusters) {
    min_num_clusters_ = min_num_clusters;
  }
  void setNumLoops(unsigned num_loops) { num_loops_ = num_loops; }
};

#endif  // SRC_KMEANS_KMEANS_BENCHMARK_H_
