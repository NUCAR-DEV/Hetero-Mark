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

#include "src/kmeans/hsa/kmeans_hsa_benchmark.h"
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include "src/kmeans/hsa/kernels.h"

void KmeansHsaBenchmark::Initialize() {
  KmeansBenchmark::Initialize();
  kmeans_kernel_compute_init(0);
  kmeans_kernel_swap_init(0);
}

void KmeansHsaBenchmark::TransposeFeatures() {
  size_t global_work = (size_t)num_points_;
  size_t local_work_size = kBlockSize;

  SNK_INIT_LPARM(lparm, 0);
  lparm->ldims[0] = local_work_size;
  lparm->gdims[0] = global_work;
  kmeans_kernel_swap(host_features_, feature_transpose_, num_points_,
                     num_features_, lparm);
}

void KmeansHsaBenchmark::UpdateMembership(unsigned num_clusters) {
  int *new_membership = new int[num_points_];
  size_t global_work = (size_t)num_points_;
  size_t local_work_size = kBlockSize;

  SNK_INIT_LPARM(lparm, 0);
  lparm->ldims[0] = local_work_size;
  lparm->gdims[0] = global_work;

  int size = 0;
  int offset = 0;
  kmeans_kernel_compute(feature_transpose_, clusters_, new_membership,
                        num_points_, num_clusters, num_features_, offset, size,
                        lparm);

  delta_ = 0;
  for (unsigned i = 0; i < num_points_; i++) {
    if (new_membership[i] != membership_[i]) {
      delta_++;
      membership_[i] = new_membership[i];
    }
  }
  delete[] new_membership;
}

void KmeansHsaBenchmark::CreateTemporaryMemory() {
  feature_transpose_ = new float[num_points_ * num_features_];
}

void KmeansHsaBenchmark::FreeTemporaryMemory() {
  if (feature_transpose_) delete[] feature_transpose_;
}

void KmeansHsaBenchmark::KmeansClustering(unsigned num_clusters) {
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

void KmeansHsaBenchmark::Clustering() {
  membership_ = new int[num_points_];
  min_rmse_ = FLT_MAX;

  // Sweep k from min to max_clusters_ to find the best number of cluster
  for (unsigned num_clusters = min_num_clusters_;
       num_clusters <= max_num_clusters_; num_clusters++) {
    if (num_clusters > num_points_) break;

    CreateTemporaryMemory();
    TransposeFeatures();
    KmeansClustering(num_clusters);

    float rmse = CalculateRMSE();
    if (rmse < min_rmse_) {
      min_rmse_ = rmse;
      best_num_clusters_ = num_clusters;
    }
    FreeTemporaryMemory();
  }

  delete[] membership_;
}

void KmeansHsaBenchmark::Run() { Clustering(); }

void KmeansHsaBenchmark::Cleanup() { KmeansBenchmark::Cleanup(); }
