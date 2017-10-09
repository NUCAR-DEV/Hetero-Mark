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

#include "src/kmeans/cl12/kmeans_cl12_benchmark.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#include <cstdlib>
#include <memory>

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
  membership_ = reinterpret_cast<int *>(new int[num_points_]);
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

  clSetKernelArg(kmeans_kernel_swap_, 0, sizeof(cl_mem),
                 reinterpret_cast<void *>(&device_features_));
  clSetKernelArg(kmeans_kernel_swap_, 1, sizeof(cl_mem),
                 reinterpret_cast<void *>(&device_features_swap_));
  clSetKernelArg(kmeans_kernel_swap_, 2, sizeof(cl_int),
                 reinterpret_cast<void *>(&num_points_));
  clSetKernelArg(kmeans_kernel_swap_, 3, sizeof(cl_int),
                 reinterpret_cast<void *>(&num_features_));

  size_t global_work_size = (size_t)num_points_;
  size_t local_work_size = kBlockSize;

  if (global_work_size % local_work_size != 0)
    global_work_size =
        (global_work_size / local_work_size + 1) * local_work_size;

  cpu_gpu_logger_->GPUOn();
  err = clEnqueueNDRangeKernel(cmd_queue_, kmeans_kernel_swap_, 1, NULL,
                               &global_work_size, &local_work_size, 0, 0, 0);
  checkOpenCLErrors(err, "ERROR: clEnqueueNDRangeKernel()");
  clFinish(cmd_queue_);
  cpu_gpu_logger_->GPUOff();

  // std::unique_ptr<float[]> trans_result(
  //     new float[num_points_ * num_features_]());
  //
  // err = clEnqueueReadBuffer(cmd_queue_, device_features_swap_, CL_TRUE, 0,
  //                           sizeof(float) * num_points_ * num_features_,
  //                           trans_result.get(), 0, 0, NULL);
}

void KmeansCl12Benchmark::KmeansClustering(unsigned num_clusters) {
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
    std::cout << "Start" << std::endl;
    UpdateMembership(num_clusters);
    std::cout << "Done updating membership" << std::endl;
    UpdateClusterCentroids(num_clusters);
    num_iteration++;
  } while ((delta_ > 0) && (num_iteration < 500));

  std::cout << "# of iterations: " << num_iteration << std::endl;
}

void KmeansCl12Benchmark::UpdateMembership(unsigned num_clusters) {
  std::unique_ptr<int[]> new_membership(new int[num_points_]());

  size_t global_work_size = (size_t)num_points_;
  size_t local_work_size = kBlockSize;

  if (global_work_size % local_work_size != 0)
    global_work_size =
        (global_work_size / local_work_size + 1) * local_work_size;

  cl_int err;
  err = clEnqueueWriteBuffer(cmd_queue_, device_clusters_, 1, 0,
                             num_clusters_ * num_features_ * sizeof(float),
                             clusters_, 0, 0, 0);
  checkOpenCLErrors(err, "ERROR: clEnqueueWriteBuffer device_clusters_");

  int size = 0;
  int offset = 0;

  clSetKernelArg(kmeans_kernel_compute_, 0, sizeof(cl_mem),
                 reinterpret_cast<void *>(&device_features_swap_));
  clSetKernelArg(kmeans_kernel_compute_, 1, sizeof(cl_mem),
                 reinterpret_cast<void *>(&device_clusters_));
  clSetKernelArg(kmeans_kernel_compute_, 2, sizeof(cl_mem),
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

  cpu_gpu_logger_->GPUOn();
  err = clEnqueueNDRangeKernel(cmd_queue_, kmeans_kernel_compute_, 1, NULL,
                               &global_work_size, &local_work_size, 0, 0, 0);
  checkOpenCLErrors(err,
                    "ERROR: clEnqueueNDRangeKernel kmeans_kernel_compute_");
  clFinish(cmd_queue_);
  cpu_gpu_logger_->GPUOff();

  err = clEnqueueReadBuffer(cmd_queue_, device_membership_, 1, 0,
                            num_points_ * sizeof(int), new_membership.get(), 0,
                            0, 0);
  checkOpenCLErrors(err, "ERROR: Memcopy Out");
  clFinish(cmd_queue_);

  cpu_gpu_logger_->CPUOn();
  delta_ = 0.0f;
  for (unsigned i = 0; i < num_points_; i++) {
    if (new_membership[i] != membership_[i]) {
      delta_++;
      membership_[i] = new_membership[i];
    }
  }
  cpu_gpu_logger_->CPUOff();
}

void KmeansCl12Benchmark::InitializeData() {}

void KmeansCl12Benchmark::Run() { 
  Clustering(); 
  cpu_gpu_logger_->Summarize();
}

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
}
