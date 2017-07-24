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

#include "src/kmeans/cl20/kmeans_cl20_benchmark.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <cstdlib>

void KmeansCl20Benchmark::Initialize() {
  KmeansBenchmark::Initialize();

  ClBenchmark::InitializeCl();

  InitializeKernels();
  InitializeBuffers();
  InitializeData();
}

void KmeansCl20Benchmark::InitializeKernels() {
  cl_int err;
  file_->open("kernels.cl");

  const char *source = file_->getSourceChar();
  program_ = clCreateProgramWithSource(context_, 1, (const char **)&source,
                                       NULL, &err);
  checkOpenCLErrors(err, "Failed to create program with source...\n");

  err = clBuildProgram(program_, 0, NULL, "-I ./ -cl-std=CL2.0", NULL, NULL);
  checkOpenCLErrors(err, "Failed to create program...\n");

  kmeans_kernel_compute_ =
      clCreateKernel(program_, "kmeans_kernel_compute", &err);
  checkOpenCLErrors(err, "Failed to create kernel kmeans_kernel_compute\n");

  kmeans_kernel_swap_ = clCreateKernel(program_, "kmeans_kernel_swap", &err);
  checkOpenCLErrors(err, "Failed to create kernel kmeans_kernel_swap\n");
}

void KmeansCl20Benchmark::InitializeBuffers() {
  size_t bytes_features = num_features_ * num_points_ * sizeof(float);
  svm_features_ = reinterpret_cast<float *>(
      clSVMAlloc(context_, CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER,
                 bytes_features, 0));
  assert(svm_features_ && "clSVMAlloc failed: svm_features_");
  svm_features_swap_ = reinterpret_cast<float *>(
      clSVMAlloc(context_, CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER,
                 bytes_features, 0));
  assert(svm_features_swap_ && "clSVMAlloc failed: svm_features_swap_");

  size_t bytes_membership = num_points_ * sizeof(int);
  svm_membership_ = reinterpret_cast<int *>(
      clSVMAlloc(context_, CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER,
                 bytes_membership, 0));
  assert(svm_membership_ && "clSVMAlloc failed: svm_membership_");
}

void KmeansCl20Benchmark::CreateTemporaryMemory() {
  size_t bytes_clusters = num_clusters_ * sizeof(int);
  svm_clusters_ = reinterpret_cast<int *>(
      clSVMAlloc(context_, CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER,
                 bytes_clusters, 0));
  assert(svm_clusters_ && "clSVMAlloc failed: svm_clusters_");
}

void KmeansCl20Benchmark::FreeTemporaryMemory() {
  if (svm_clusters_) clSVMFree(context_, svm_clusters_);
}

void KmeansCl20Benchmark::Clustering() {
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
void KmeansCl20Benchmark::TransposeFeatures() {
  size_t bytes_features = num_points_ * num_features_ * sizeof(float);
  memcpy(svm_features_, svm_features_, bytes_features);

  clSetKernelArgSVMPointer(kmeans_kernel_swap_, 0, svm_features_);
  clSetKernelArgSVMPointer(kmeans_kernel_swap_, 1, svm_features_swap_);
  clSetKernelArg(kmeans_kernel_swap_, 2, sizeof(cl_int),
                 reinterpret_cast<void *>(&num_points_));
  clSetKernelArg(kmeans_kernel_swap_, 3, sizeof(cl_int),
                 reinterpret_cast<void *>(&num_features_));

  size_t global_work_size = (size_t)num_points_;
  size_t local_work_size = kBlockSize;

  if (global_work_size % local_work_size != 0)
    global_work_size =
        (global_work_size / local_work_size + 1) * local_work_size;

  cl_int err;
  err = clEnqueueNDRangeKernel(cmd_queue_, kmeans_kernel_swap_, 1, NULL,
                               &global_work_size, &local_work_size, 0, 0, 0);
  checkOpenCLErrors(err, "ERROR: clEnqueueNDRangeKernel()");
}

void KmeansCl20Benchmark::KmeansClustering(unsigned num_clusters) {
  int num_iteration = 0;

  // Sanity check: avoid a cluster without points
  if (num_clusters > num_points_) {
    std::cerr << "# of clusters < # of points" << std::endl;
    exit(-1);
  }

  InitializeClusters(num_clusters);
  InitializeMembership();

  // Iterate until convergence
  do {
    UpdateMembership(num_clusters);
    UpdateClusterCentroids(num_clusters);
    num_iteration++;
  } while ((delta_ > 0) && (num_iteration < 500));

  std::cout << "# of iterations: " << num_iteration << std::endl;
}

void KmeansCl20Benchmark::InitializeClusters(unsigned num_clusters) {
  size_t bytes_clusters = num_points_ * num_features_ * sizeof(int);
  svm_clusters_ = reinterpret_cast<int *>(
      clSVMAlloc(context_, CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER,
                 bytes_clusters, 0));

  for (unsigned i = 1; i < num_clusters; i++)
    svm_clusters_[i] = svm_clusters_[i - 1] + num_features_;

  for (unsigned i = 0; i < num_clusters * num_features_; i++)
    svm_clusters_[i] = svm_features_[i];
}

void KmeansCl20Benchmark::InitializeMembership() {
  for (unsigned i = 0; i < num_points_; i++) svm_membership_[i] = -1;
}

void KmeansCl20Benchmark::UpdateMembership(unsigned num_clusters) {
  size_t bytes_membership = num_points_ * sizeof(int);
  int *new_membership = reinterpret_cast<int *>(
      clSVMAlloc(context_, CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER,
                 bytes_membership, 0));

  for (unsigned i = 0; i < num_points_; i++)
    new_membership[i] = svm_membership_[i];

  size_t global_work_size = (size_t)num_points_;
  size_t local_work_size = kBlockSize;

  if (global_work_size % local_work_size != 0)
    global_work_size =
        (global_work_size / local_work_size + 1) * local_work_size;

  int size = 0;
  int offset = 0;

  clSetKernelArgSVMPointer(kmeans_kernel_compute_, 0, svm_features_swap_);
  clSetKernelArgSVMPointer(kmeans_kernel_compute_, 1, svm_clusters_);
  clSetKernelArgSVMPointer(kmeans_kernel_compute_, 2, new_membership);
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
  cl_int err;
  err = clEnqueueNDRangeKernel(cmd_queue_, kmeans_kernel_compute_, 1, NULL,
                               &global_work_size, &local_work_size, 0, 0, 0);
  checkOpenCLErrors(err,
                    "ERROR: clEnqueueNDRangeKernel kmeans_kernel_compute_");

  clFinish(cmd_queue_);

  delta_ = 0.0f;
  for (unsigned i = 0; i < num_points_; i++) {
    if (new_membership[i] != svm_membership_[i]) {
      delta_++;
      svm_membership_[i] = new_membership[i];
    }
  }

  clSVMFree(context_, new_membership);
}

void KmeansCl20Benchmark::UpdateClusterCentroids(unsigned num_clusters) {
  // Allocate space for and initialize new_centers_len and new_centers
  int *member_count = new int[num_clusters]();

  // Clean up svm_clusters_
  for (unsigned i = 0; i < num_clusters * num_features_; ++i)
    svm_clusters_[i] = 0;

  // Calculate sum
  for (unsigned i = 0; i < num_points_; i++) {
    for (unsigned j = 0; j < num_features_; j++) {
      unsigned index_feature = i * num_features_ + j;
      unsigned index_cluster = svm_membership_[i] * num_features_ + j;
      svm_clusters_[index_cluster] += svm_features_[index_feature];
    }
    member_count[svm_membership_[i]]++;
  }

  // For each cluster, devide by the number of points in the cluster
  for (unsigned i = 0; i < num_clusters; i++) {
    for (unsigned j = 0; j < num_features_; j++) {
      unsigned index = i * num_features_ + j;
      if (member_count[i]) svm_clusters_[index] /= member_count[i];
    }
  }

  // Free space
  delete[] member_count;
}

float KmeansCl20Benchmark::CalculateRMSE() {
  float mean_square_error = 0;
  for (unsigned i = 0; i < num_points_; i++) {
    float distance_square = 0;
    for (unsigned j = 0; j < num_features_; j++) {
      unsigned index_feature = i * num_features_ + j;
      unsigned index_cluster = svm_membership_[i] * num_features_ + j;
      distance_square +=
          pow((svm_features_[index_feature] - svm_clusters_[index_cluster]), 2);
    }
    mean_square_error += distance_square;
  }
  mean_square_error /= num_points_;

  return sqrt(mean_square_error);
}

void KmeansCl20Benchmark::InitializeData() {}

void KmeansCl20Benchmark::Run() { Clustering(); }

void KmeansCl20Benchmark::Cleanup() {
  KmeansBenchmark::Cleanup();

  cl_int ret;
  ret = clReleaseKernel(kmeans_kernel_swap_);
  ret = clReleaseKernel(kmeans_kernel_compute_);
  ret = clReleaseProgram(program_);
  checkOpenCLErrors(ret, "ERROR: clReleaseKernel/Program failed");

  // OTHER_CLEANUPS
  clSVMFree(context_, svm_features_);
  clSVMFree(context_, svm_features_swap_);
  clSVMFree(context_, svm_membership_);
}
