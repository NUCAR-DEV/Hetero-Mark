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
    this list of conditions and the following disclaimers.
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

#include "src/kmeans/hc/kmeans_hc_benchmark.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <hcc/hc.hpp>

void KmeansHcBenchmark::Initialize() { KmeansBenchmark::Initialize(); }

void KmeansHcBenchmark::TransposeFeatures() {}

void KmeansHcBenchmark::UpdateMembership(unsigned num_clusters) {
  int *new_membership = new int[num_points_];
  hc::array_view<int, 1> av_membership(num_points_, new_membership);
  hc::array_view<float, 1> av_features(num_points_ * num_features_,
                                       host_features_);
  hc::array_view<float, 1> av_clusters(num_clusters * num_features_, clusters_);

  unsigned num_features = num_features_;

  hc::extent<1> ex(num_points_);

  parallel_for_each(ex, [=](hc::index<1> i)[[hc]] {
    float min_dist = FLT_MAX;
    int index = 0;
    for (uint32_t j = 0; j < num_clusters; j++) {
      float dist = 0;
      for (uint32_t k = 0; k < num_features; k++) {
        float diff = av_features[i[0] * num_features + k] -
                     av_clusters[j * num_features + k];
        dist += diff * diff;
      }

      if (dist < min_dist) {
        min_dist = dist;
        index = j;
      }
    }
    av_membership[i] = index;
  });
  av_membership.synchronize();

  delta_ = 0;
  for (uint32_t i = 0; i < num_points_; i++) {
    if (new_membership[i] != membership_[i]) {
      delta_++;
      membership_[i] = new_membership[i];
    }
  }

  delete[] new_membership;
}

void KmeansHcBenchmark::CreateTemporaryMemory() {}

void KmeansHcBenchmark::FreeTemporaryMemory() {}

void KmeansHcBenchmark::KmeansClustering(unsigned num_clusters) {
  unsigned num_iteration = 0;

  // that would guarantee a cluster without points
  if (num_clusters > num_points_) {
    printf("# of clusters cannot be less than # of points\n");
    exit(-1);
  }

  InitializeClusters(num_clusters);
  InitializeMembership();

  // iterate until converge
  do {
    UpdateMembership(num_clusters);
    UpdateClusterCentroids(num_clusters);
    num_iteration++;
  } while ((delta_ > 0) && (num_iteration < 500));

  printf("iterated %d times\n", num_iteration);
}

void KmeansHcBenchmark::Clustering() {
  min_rmse_ = FLT_MAX;
  membership_ = new int[num_points_];
  // Sweep k from min to max_clusters_ to find the best number of cluster
  for (unsigned num_clusters = min_num_clusters_;
       num_clusters <= max_num_clusters_; num_clusters++) {
    if (num_clusters > num_points_) break;

    // CreateTemporaryMemory();
    // TransposeFeatures();
    KmeansClustering(num_clusters);

    float rmse = CalculateRMSE();
    if (rmse < min_rmse_) {
      min_rmse_ = rmse;
      best_num_clusters_ = num_clusters;
    }
    // FreeTemporaryMemory();
  }
  delete[] membership_;
}

void KmeansHcBenchmark::Run() { Clustering(); }

void KmeansHcBenchmark::Cleanup() { KmeansBenchmark::Cleanup(); }
