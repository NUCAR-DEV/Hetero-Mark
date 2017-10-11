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

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <cstdlib>
#include "hip/hip_runtime.h"
#include "src/kmeans/hip/kmeans_hip_benchmark.h"

__global__ void kmeans_swap_hip(hipLaunchParm lp, float *feature,
                                float *feature_swap, int npoints,
                                int nfeatures) {
  uint tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  if (tid >= npoints) return;

  for (int i = 0; i < nfeatures; i++)
    feature_swap[i * npoints + tid] = feature[tid * nfeatures + i];
}

__global__ void kmeans_compute_hip(hipLaunchParm lp, float *feature,
                                   float *clusters, int *membership,
                                   int npoints, int nclusters, int nfeatures,
                                   int offset, int size) {
  int point_id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  if (point_id >= npoints) return;

  int index = 0;
  if (point_id < npoints) {
    float min_dist = FLT_MAX;
    for (int i = 0; i < nclusters; i++) {
      float dist = 0;
      float ans = 0;
      for (int l = 0; l < nfeatures; l++) {
        ans += (feature[l * npoints + point_id] - clusters[i * nfeatures + l]) *
               (feature[l * npoints + point_id] - clusters[i * nfeatures + l]);
      }

      dist = ans;
      if (dist < min_dist) {
        min_dist = dist;
        index = i;
      }
    }
    membership[point_id] = index;
  }

  return;
}

void KmeansHipBenchmark::Initialize() {
  KmeansBenchmark::Initialize();

  InitializeBuffers();
}

void KmeansHipBenchmark::InitializeBuffers() {
  hipMalloc(&device_membership_, num_points_ * sizeof(int));
  hipMalloc(&device_features_, num_points_ * num_features_ * sizeof(float));
  hipMalloc(&device_features_swap_,
            num_points_ * num_features_ * sizeof(float));
}

void KmeansHipBenchmark::CreateTemporaryMemory() {
  hipMalloc(&device_clusters_, num_clusters_ * num_features_ * sizeof(float));
}

void KmeansHipBenchmark::FreeTemporaryMemory() { hipFree(device_clusters_); }

void KmeansHipBenchmark::Clustering() {
  min_rmse_ = FLT_MAX;
  membership_ = new int[num_points_];

  // Sweep k from min to max_clusters_ to find the best number of clusters
  for (num_clusters_ = min_num_clusters_; num_clusters_ <= max_num_clusters_;
       num_clusters_++) {
    // Sanity check: cannot have more clusters than points
    if (num_clusters_ > num_points_) break;

    CreateTemporaryMemory();
    TransposeFeatures();
    KmeansClustering(num_clusters_);

    float rmse = CalculateRMSE();
    if (rmse < min_rmse_) {
      min_rmse_ = rmse;
      best_num_clusters_ = num_clusters_;
    }
    FreeTemporaryMemory();
  }

  delete[] membership_;
}

void KmeansHipBenchmark::TransposeFeatures() {
  hipMemcpy(device_features_, host_features_,
            num_points_ * num_features_ * sizeof(float), hipMemcpyHostToDevice);

  dim3 block_size(64);
  dim3 grid_size((num_points_ + block_size.x - 1) / block_size.x);

  cpu_gpu_logger_->GPUOn();
  hipLaunchKernel(HIP_KERNEL_NAME(kmeans_swap_hip), dim3(grid_size),
                  dim3(block_size), 0, 0, device_features_,
                  device_features_swap_, num_points_, num_features_);
  hipDeviceSynchronize();
  cpu_gpu_logger_->GPUOff();
}

void KmeansHipBenchmark::KmeansClustering(unsigned num_clusters) {
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

void KmeansHipBenchmark::UpdateMembership(unsigned num_clusters) {
  int *new_membership = new int[num_points_];

  dim3 block_size(64);
  dim3 grid_size((num_points_ + block_size.x - 1) / block_size.x);

  hipMemcpy(device_clusters_, clusters_,
            num_clusters_ * num_features_ * sizeof(float),
            hipMemcpyHostToDevice);

  int size = 0;
  int offset = 0;

  cpu_gpu_logger_->GPUOn();
  hipLaunchKernel(HIP_KERNEL_NAME(kmeans_compute_hip), dim3(grid_size),
                  dim3(block_size), 0, 0, device_features_swap_,
                  device_clusters_, device_membership_, num_points_,
                  num_clusters_, num_features_, offset, size);
  hipDeviceSynchronize();
  cpu_gpu_logger_->GPUOff();

  hipMemcpy(new_membership, device_membership_, num_points_ * sizeof(int),
            hipMemcpyDeviceToHost);

  cpu_gpu_logger_->CPUOn();
  delta_ = 0.0f;
  for (unsigned int i = 0; i < num_points_; i++) {
    /* printf("number %d, merbership %d\n", i, new_membership[i]); */
    if (new_membership[i] != membership_[i]) {
      delta_++;
      membership_[i] = new_membership[i];
    }
  }
  cpu_gpu_logger_->CPUOff();
}

void KmeansHipBenchmark::Run() { 
  Clustering(); 
  cpu_gpu_logger_->Summarize();
}

void KmeansHipBenchmark::Cleanup() {
  hipFree(device_features_);
  hipFree(device_features_swap_);
  hipFree(device_membership_);
}
