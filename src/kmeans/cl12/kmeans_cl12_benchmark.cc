/* Copyright (c) 2015 Northeastern University
 * All rights reserved.
 *
 * Developed by:Northeastern University Computer Architecture Research (NUCAR)
 * Group, Northeastern University, http://www.ece.neu.edu/groups/nucar/
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 *  with the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense, and/
 * or sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *   Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimers. Redistributions in binary
 *   form must reproduce the above copyright notice, this list of conditions and
 *   the following disclaimers in the documentation and/or other materials
 *   provided with the distribution. Neither the names of NUCAR, Northeastern
 *   University, nor the names of its contributors may be used to endorse or
 *   promote products derived from this Software without specific prior written
 *   permission.
 *
 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 *   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 *   DEALINGS WITH THE SOFTWARE.
 */

#include <math.h>
#include <string.h>
#include <stdio.h>
#include <cstdlib>
#include "src/kmeans/cl12/kmeans_cl12_benchmark.h"

void KmeansCl12Benchmark::Initialize() {
  KmeansBenchmark::Initialize();

  ClBenchmark::InitializeCl();

  InitializeKernels();
  InitializeBuffers();
  InitializeData();
}

void KmeansCl12Benchmark::InitializeKernels() {
  cl_int err;
  file_->open("kernels.cl");

  const char *source = file_->getSourceChar();
  program_ = clCreateProgramWithSource(context_, 1, (const char **)&source,
                                       NULL, &err);
  checkOpenCLErrors(err, "Failed to create program with source...\n");

  err = clBuildProgram(program_, 0, NULL, NULL, NULL, NULL);
  checkOpenCLErrors(err, "Failed to create program...\n");

  kmeans_kernel_compute_ =
      clCreateKernel(program_, "kmeans_kernel_compute", &err);
  checkOpenCLErrors(err, "Failed to create kernel XXX\n");

  kmeans_kernel_swap_ = clCreateKernel(program_, "kmeans_kernel_swap", &err);
  checkOpenCLErrors(err, "Failed to create kernel XXX\n");
}

void KmeansCl12Benchmark::InitializeBuffers() {
  cl_int err;

  // Create device buffers
  device_features_ =
      clCreateBuffer(context_, CL_MEM_READ_WRITE,
                     num_points_ * num_features_ * sizeof(float), NULL, &err);
  checkOpenCLErrors(err, "clCreateBuffer d_feature failed");

  device_features_swap_ =
      clCreateBuffer(context_, CL_MEM_READ_WRITE,
                     num_points_ * num_features_ * sizeof(float), NULL, &err);
  checkOpenCLErrors(err, "clCreateBuffer device_features_swap failed");

  device_membership_ = clCreateBuffer(context_, CL_MEM_READ_WRITE,
                                      num_points_ * sizeof(int), NULL, &err);
  checkOpenCLErrors(err, "clCreateBuffer d_membership failed");

  // Create host buffers
  host_membership_ = reinterpret_cast<int *>(new int[num_points_]);
}

void KmeansCl12Benchmark::CreateTemporaryMemory() {
  cl_int err;
  device_clusters_ =
      clCreateBuffer(context_, CL_MEM_READ_WRITE,
                     num_clusters_ * num_features_ * sizeof(float), NULL, &err);
  checkOpenCLErrors(err, "clCreateBuffer d_cluster failed");
}

void KmeansCl12Benchmark::FreeTemporaryMemory() {
  cl_int err;
  err = clReleaseMemObject(device_clusters_);
  checkOpenCLErrors(err, "clCreateBuffer d_cluster failed");
}

void KmeansCl12Benchmark::Clustering() {
  min_rmse_ = FLT_MAX;

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
}

void KmeansCl12Benchmark::TransposeFeatures() {
  cl_int err;

  // Copy feature from host to device
  err = clEnqueueWriteBuffer(cmd_queue_, device_features_, 1, 0,
                             num_points_ * num_features_ * sizeof(float),
                             host_features_, 0, 0, 0);
  checkOpenCLErrors(err, "ERROR: clEnqueueWriteBuffer device_features_");

  clSetKernelArg(kmeans_kernel_swap_, 0, sizeof(void *),
                 reinterpret_cast<void *>(&device_features_));
  clSetKernelArg(kmeans_kernel_swap_, 1, sizeof(void *),
                 reinterpret_cast<void *>(&device_features_swap_));
  clSetKernelArg(kmeans_kernel_swap_, 2, sizeof(cl_int),
                 reinterpret_cast<void *>(&num_points_));
  clSetKernelArg(kmeans_kernel_swap_, 3, sizeof(cl_int),
                 reinterpret_cast<void *>(&num_features_));

  size_t global_work_size = (size_t)num_points_;
  size_t local_work_size = BLOCK_SIZE;

  if (global_work_size % local_work_size != 0)
    global_work_size =
        (global_work_size / local_work_size + 1) * local_work_size;

  err = clEnqueueNDRangeKernel(cmd_queue_, kmeans_kernel_swap_, 1, NULL,
                               &global_work_size, &local_work_size, 0, 0, 0);
  checkOpenCLErrors(err, "ERROR: clEnqueueNDRangeKernel()");
}

void KmeansCl12Benchmark::KmeansClustering(unsigned num_clusters) {
  int num_iteration = 0;

  // Sanity check: avoid a cluster without points
  if (num_clusters > num_points_) {
    std::cerr << "# of clusters < # of points" << std::endl;
    exit(-1);
  }

  InitializeHostClusters(num_clusters);
  InitializeHostMembership();

  // Iterate until convergence
  do {
    UpdateMembership(num_clusters);
    UpdateClusterCentroids(num_clusters);
    num_iteration++;
  } while ((delta_ > 0) && (num_iteration < 500));

  delete[] host_clusters_;

  std::cout << "# of iterations: " << num_iteration;
}

void KmeansCl12Benchmark::InitializeHostClusters(unsigned num_clusters) {
  host_clusters_ = new float[num_clusters * num_features_];
  for (unsigned i = 1; i < num_clusters; i++)
    host_clusters_[i] = host_clusters_[i - 1] + num_features_;

  for (unsigned i = 0; i < num_clusters * num_features_; i++)
    host_clusters_[i] = host_features_[i];
}

void KmeansCl12Benchmark::InitializeHostMembership() {
  for (unsigned i = 0; i < num_points_; i++) {
    host_membership_[i] = -1;
  }
}

void KmeansCl12Benchmark::UpdateMembership(unsigned num_clusters) {
  int *new_membership = new int[num_points_];
  size_t global_work_size = (size_t)num_points_;
  size_t local_work_size = BLOCK_SIZE;

  if (global_work_size % local_work_size != 0)
    global_work_size =
        (global_work_size / local_work_size + 1) * local_work_size;

  cl_int err;
  err = clEnqueueWriteBuffer(cmd_queue_, device_clusters_, 1, 0,
                             num_clusters_ * num_features_ * sizeof(float),
                             host_clusters_, 0, 0, 0);
  checkOpenCLErrors(err, "ERROR: clEnqueueWriteBuffer device_clusters_");

  int size = 0;
  int offset = 0;

  clSetKernelArg(kmeans_kernel_compute_, 0, sizeof(void *),
                 reinterpret_cast<void *>(&device_features_swap_));
  clSetKernelArg(kmeans_kernel_compute_, 1, sizeof(void *),
                 reinterpret_cast<void *>(&device_clusters_));
  clSetKernelArg(kmeans_kernel_compute_, 2, sizeof(void *),
                 reinterpret_cast<void *>(&device_membership_));
  clSetKernelArg(kmeans_kernel_compute_, 3, sizeof(cl_int),
                 reinterpret_cast<void *>(&num_points_));
  clSetKernelArg(kmeans_kernel_compute_, 4, sizeof(cl_int),
                 reinterpret_cast<void *>(&num_clusters));
  clSetKernelArg(kmeans_kernel_compute_, 5, sizeof(cl_int),
                 reinterpret_cast<void *>(&num_features_));
  clSetKernelArg(kmeans_kernel_compute_, 6, sizeof(cl_int),
                 reinterpret_cast<void *>(&offset));
  clSetKernelArg(kmeans_kernel_compute_, 7, sizeof(cl_int),
                 reinterpret_cast<void *>(&size));

  err = clEnqueueNDRangeKernel(cmd_queue_, kmeans_kernel_compute_, 1, NULL,
                               &global_work_size, &local_work_size, 0, 0, 0);
  checkOpenCLErrors(err,
                    "ERROR: clEnqueueNDRangeKernel kmeans_kernel_compute_");

  clFinish(cmd_queue_);

  err = clEnqueueReadBuffer(cmd_queue_, device_membership_, 1, 0,
                            num_points_ * sizeof(int), new_membership, 0, 0, 0);
  checkOpenCLErrors(err, "ERROR: Memcopy Out");

  delta_ = 0.0f;
  for (unsigned i = 0; i < num_points_; i++) {
    if (new_membership[i] != host_membership_[i]) {
      delta_++;
      host_membership_[i] = new_membership[i];
    }
  }

  delete[] new_membership;
}

void KmeansCl12Benchmark::UpdateClusterCentroids(unsigned num_clusters) {
  // Allocate space for and initialize new_centers_len and new_centers
  int *member_count = new int[num_clusters]();

  // Clean up host_clusters_
  for (unsigned i = 0; i < num_clusters * num_features_; ++i)
    host_clusters_[i] = 0;

  // Calculate sum
  for (unsigned i = 0; i < num_points_; i++) {
    for (unsigned j = 0; j < num_features_; j++) {
      unsigned index_feature = i * num_features_ + j;
      unsigned index_cluster = host_membership_[i] * num_features_ + j;
      host_clusters_[index_cluster] += host_features_[index_feature];
    }
    member_count[host_membership_[i]]++;
  }

  // For each cluster, devide by the number of points in the cluster
  for (unsigned i = 0; i < num_clusters; i++) {
    for (unsigned j = 0; j < num_features_; j++) {
      unsigned index = i * num_features_ + j;
      if (member_count[i]) host_clusters_[index] /= member_count[i];
    }
  }

  delete[] member_count;
}

float KmeansCl12Benchmark::CalculateRMSE() {
  float mean_square_error = 0;
  for (unsigned i = 0; i < num_points_; i++) {
    float distance_square = 0;
    for (unsigned j = 0; j < num_features_; j++) {
      unsigned index_feature = i * num_features_ + j;
      unsigned index_cluster = host_membership_[i] * num_features_ + j;
      distance_square += pow(
          (host_features_[index_feature] - host_clusters_[index_cluster]), 2);
    }
    mean_square_error += distance_square;
  }
  mean_square_error /= num_points_;
  return sqrt(mean_square_error);
}

void KmeansCl12Benchmark::InitializeData() {}

void KmeansCl12Benchmark::Run() { Clustering(); }

void KmeansCl12Benchmark::Cleanup() {
  KmeansBenchmark::Cleanup();

  cl_int ret;
  ret = clReleaseKernel(kmeans_kernel_swap_);
  ret = clReleaseKernel(kmeans_kernel_compute_);
  ret = clReleaseProgram(program_);
  checkOpenCLErrors(ret, "Release kernerl and program.\n");

  ret = clReleaseMemObject(device_features_);
  ret |= clReleaseMemObject(device_features_swap_);
  ret |= clReleaseMemObject(device_membership_);
  checkOpenCLErrors(ret, "Release mem object.\n");

  delete[] host_membership_;
}
