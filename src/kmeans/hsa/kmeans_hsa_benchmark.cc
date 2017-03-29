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
#include <cstring>
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
  kmeans_kernel_swap(d_features_, d_features_transpose_, num_points_,
                     num_features_, lparm);

}

void KmeansHsaBenchmark::UpdateMembership(unsigned num_clusters) {
  int *new_membership = reinterpret_cast<int *>(
      malloc_global(num_points_ * sizeof(int)));
  size_t global_work = (size_t)num_points_;
  size_t local_work_size = kBlockSize;

  SNK_INIT_LPARM(lparm, 0);
  lparm->ldims[0] = local_work_size;
  lparm->gdims[0] = global_work;

  int size = 0;
  int offset = 0;
  kmeans_kernel_compute(d_features_transpose_, d_clusters_, new_membership,
                        num_points_, num_clusters, num_features_, offset, size,
                        lparm);

  delta_ = 0;
  for (unsigned i = 0; i < num_points_; i++) {
    if (new_membership[i] != d_membership_[i]) {
      delta_++;
      d_membership_[i] = new_membership[i];
    }
  }
  free_global(new_membership);
}

void KmeansHsaBenchmark::InitializeClustersGPU(unsigned num_clusters) {
  d_clusters_ = reinterpret_cast<float *>(
      malloc_global(num_clusters * num_features_ * sizeof(float)));

  for (unsigned i = 0; i < num_clusters * num_features_; i++)
    d_clusters_[i] = host_features_[i];
}

void KmeansHsaBenchmark::InitializeMembershipGPU() {
  for (unsigned i = 0; i < num_points_; i++) {
    d_membership_[i] = -1;
  }
}


void KmeansHsaBenchmark::CreateTemporaryMemory() {
  d_features_transpose_ = reinterpret_cast<float *>(
      malloc_global(num_points_ * num_features_ *sizeof(float)));
  d_features_ = reinterpret_cast<float *>(
      malloc_global(num_points_ * num_features_ * sizeof(float)));
  d_membership_ = reinterpret_cast<int *>(
      malloc_global(num_points_ * sizeof(int)));

  memcpy(d_features_, host_features_, 
      num_points_ * num_features_ * sizeof(float));
}

void KmeansHsaBenchmark::FreeTemporaryMemory() {
  free_global(d_features_transpose_);
  free_global(d_features_);
  free_global(d_membership_);
  free_global(d_clusters_);
}

void KmeansHsaBenchmark::KmeansClustering(unsigned num_clusters) {
  unsigned num_iteration = 0;

  // that would guarantee a cluster without points
  if (num_clusters > num_points_) {
    printf("# of clusters cannot be less than # of points\n");
    exit(-1);
  }

  InitializeClustersGPU(num_clusters);
  InitializeMembershipGPU();

  // iterate until converge
  do {
    UpdateMembership(num_clusters);
    UpdateClusterCentroidsGPU(num_clusters);
    num_iteration++;
  } while ((delta_ > 0) && (num_iteration < 500));

  printf("iterated %d times\n", num_iteration);
}

void KmeansHsaBenchmark::UpdateClusterCentroidsGPU(unsigned num_clusters) {
  // Allocate space for and initialize new_centers_len and new_centers
  int *member_count = new int[num_clusters]();

  // Clean up clusters_
  for (unsigned i = 0; i < num_clusters * num_features_; i++) {
    d_clusters_[i] = 0;
  }

  // Calculate sum
  for (unsigned i = 0; i < num_points_; i++) {
    for (unsigned j = 0; j < num_features_; j++) {
      unsigned index_feature = i * num_features_ + j;
      unsigned index_cluster = d_membership_[i] * num_features_ + j;
      d_clusters_[index_cluster] += host_features_[index_feature];
    }
    member_count[d_membership_[i]]++;
  }

  // For each cluster, divide by the number of points in the cluster
  for (unsigned i = 0; i < num_clusters; i++) {
    for (unsigned j = 0; j < num_features_; j++) {
      unsigned index = i * num_features_ + j;
      if (member_count[i]) d_clusters_[index] /= member_count[i];
    }
  }

  // DumpClusterCentroids(num_clusters);
  delete[] member_count;
}

float KmeansHsaBenchmark::CalculateRMSEGPU() {
  float mean_square_error = 0;
  for (unsigned i = 0; i < num_points_; i++) {
    float distance_square = 0;
    for (unsigned j = 0; j < num_features_; j++) {
      unsigned index_feature = i * num_features_ + j;
      unsigned index_cluster = d_membership_[i] * num_features_ + j;
      distance_square +=
          pow((host_features_[index_feature] - d_clusters_[index_cluster]), 2);
    }
    mean_square_error += distance_square;
  }
  mean_square_error /= num_points_;
  return sqrt(mean_square_error);
}



void KmeansHsaBenchmark::Clustering() {
  min_rmse_ = FLT_MAX;

  // Sweep k from min to max_clusters_ to find the best number of cluster
  for (unsigned num_clusters = min_num_clusters_;
       num_clusters <= max_num_clusters_; num_clusters++) {
    if (num_clusters > num_points_) break;

    CreateTemporaryMemory();
    TransposeFeatures();
    KmeansClustering(num_clusters);

    float rmse = CalculateRMSEGPU();
    if (rmse < min_rmse_) {
      min_rmse_ = rmse;
      best_num_clusters_ = num_clusters;
    }
    FreeTemporaryMemory();
  }

}

void KmeansHsaBenchmark::Run() { Clustering(); }

void KmeansHsaBenchmark::Cleanup() { KmeansBenchmark::Cleanup(); }
