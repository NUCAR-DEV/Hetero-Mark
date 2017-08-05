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

#include "src/pr/cl20/pr_cl20_benchmark.h"
#include <stdio.h>
#include <string.h>
#include <cstdlib>

void PrCl20Benchmark::Initialize() {
  PrBenchmark::Initialize();

  ClBenchmark::InitializeCl();

  InitializeKernels();
  InitializeBuffers();
}

void PrCl20Benchmark::InitializeKernels() {
  cl_int err;
  file_->open("kernels.cl");

  const char *source = file_->getSourceChar();
  program_ = clCreateProgramWithSource(context_, 1, (const char **)&source,
                                       NULL, &err);
  checkOpenCLErrors(err, "Failed to create program with source...\n");

  err =
      clBuildProgram(program_, 1, &device_, "-I ./ -cl-std=CL2.0", NULL, NULL);
  checkOpenCLErrors(err, "Failed to create program...\n");

  pr_kernel_ = clCreateKernel(program_, "PageRankUpdateGpu", &err);
  checkOpenCLErrors(err, "Failed to create kernel PageRankUpdateGpu\n");
}

void PrCl20Benchmark::InitializeBuffers() {
  dev_page_rank_ = reinterpret_cast<float *>(
      clSVMAlloc(context_, CL_MEM_READ_WRITE, num_nodes_ * sizeof(float), 0));

  dev_page_rank_temp_ = reinterpret_cast<float *>(
      clSVMAlloc(context_, CL_MEM_READ_WRITE, num_nodes_ * sizeof(float), 0));

  dev_row_offsets_ = reinterpret_cast<uint32_t *>(clSVMAlloc(
      context_, CL_MEM_READ_ONLY, (num_nodes_ + 1) * sizeof(uint32_t), 0));

  dev_column_numbers_ = reinterpret_cast<uint32_t *>(clSVMAlloc(
      context_, CL_MEM_READ_ONLY, num_connections_ * sizeof(uint32_t), 0));

  dev_values_ = reinterpret_cast<float *>(clSVMAlloc(
      context_, CL_MEM_READ_ONLY, num_connections_ * sizeof(float), 0));
}

void PrCl20Benchmark::CopyDataToDevice() {
  cl_int err;

  err = clEnqueueSVMMap(cmd_queue_, CL_TRUE, CL_MAP_WRITE, dev_page_rank_,
                        num_nodes_ * sizeof(float), 0, NULL, NULL);
  checkOpenCLErrors(err, "Failed to map SVM buffer page rank");
  for (uint32_t i = 0; i < num_nodes_; i++) {
    dev_page_rank_[i] = 1.0 / num_nodes_;
  }
  err = clEnqueueSVMUnmap(cmd_queue_, dev_page_rank_, 0, NULL, NULL);
  checkOpenCLErrors(err, "Failed to unmap SVM buffer page rank");

  err = clEnqueueSVMMap(cmd_queue_, CL_TRUE, CL_MAP_WRITE, dev_row_offsets_,
                        (num_nodes_ + 1) * sizeof(uint32_t), 0, NULL, NULL);
  checkOpenCLErrors(err, "Failed to map svm buffer");
  memcpy(dev_row_offsets_, row_offsets_, (num_nodes_ + 1) * sizeof(uint32_t));
  err = clEnqueueSVMUnmap(cmd_queue_, dev_row_offsets_, 0, NULL, NULL);
  checkOpenCLErrors(err, "Failed to unmap SVM buffer page rank");

  err = clEnqueueSVMMap(cmd_queue_, CL_TRUE, CL_MAP_WRITE, dev_column_numbers_,
                        num_connections_ * sizeof(uint32_t), 0, NULL, NULL);
  checkOpenCLErrors(err, "Failed to map svm buffer");
  memcpy(dev_column_numbers_, column_numbers_,
         num_connections_ * sizeof(uint32_t));
  err = clEnqueueSVMUnmap(cmd_queue_, dev_column_numbers_, 0, NULL, NULL);
  checkOpenCLErrors(err, "Failed to unmap SVM buffer page rank");

  err = clEnqueueSVMMap(cmd_queue_, CL_TRUE, CL_MAP_WRITE, dev_values_,
                        num_connections_ * sizeof(float), 0, NULL, NULL);
  checkOpenCLErrors(err, "Failed to map svm buffer");
  memcpy(dev_values_, values_, num_connections_ * sizeof(float));
  err = clEnqueueSVMUnmap(cmd_queue_, dev_values_, 0, NULL, NULL);
  checkOpenCLErrors(err, "Failed to unmap SVM buffer page rank");

  clFinish(cmd_queue_);
}

void PrCl20Benchmark::CopyDataBackFromDevice(float *buffer) {
  cl_int err;

  err = clEnqueueSVMMap(cmd_queue_, CL_TRUE, CL_MAP_READ, buffer,
                        num_nodes_ * sizeof(float), 0, NULL, NULL);
  checkOpenCLErrors(err, "Failed to map svm buffer");

  memcpy(page_rank_, buffer, num_nodes_ * sizeof(float));
}

void PrCl20Benchmark::Run() {
  CopyDataToDevice();

  cl_int err;

  err = clSetKernelArg(pr_kernel_, 0, sizeof(uint32_t), &num_nodes_);
  checkOpenCLErrors(err, "Failed to set kernel argument 0");

  err = clSetKernelArgSVMPointer(pr_kernel_, 1, dev_row_offsets_);
  checkOpenCLErrors(err, "Failed to set kernel argument 1");

  err = clSetKernelArgSVMPointer(pr_kernel_, 2, dev_column_numbers_);
  checkOpenCLErrors(err, "Failed to set kernel argument 2");

  err = clSetKernelArgSVMPointer(pr_kernel_, 3, dev_values_);
  checkOpenCLErrors(err, "Failed to set kernel argument 3");

  err = clSetKernelArg(pr_kernel_, 4, sizeof(float) * 64, NULL);
  checkOpenCLErrors(err, "Failed to set kernel argument 4");

  uint32_t i;
  for (i = 0; i < max_iteration_; i++) {
    if (i % 2 == 0) {
      err = clSetKernelArgSVMPointer(pr_kernel_, 5, dev_page_rank_);
      checkOpenCLErrors(err, "Failed to set kernel argument 5");

      err = clSetKernelArgSVMPointer(pr_kernel_, 6, dev_page_rank_temp_);
      checkOpenCLErrors(err, "Failed to set kernel argument 6");
    } else {
      err = clSetKernelArgSVMPointer(pr_kernel_, 5, dev_page_rank_temp_);
      checkOpenCLErrors(err, "Failed to set kernel argument 5");

      err = clSetKernelArgSVMPointer(pr_kernel_, 6, dev_page_rank_);
      checkOpenCLErrors(err, "Failed to set kernel argument 6");
    }

    size_t global_work_size[] = {num_nodes_ * 64};
    size_t local_work_size[] = {64};
    err = clEnqueueNDRangeKernel(cmd_queue_, pr_kernel_, 1, NULL,
                                 global_work_size, local_work_size, 0, NULL,
                                 NULL);
    checkOpenCLErrors(err, "Failed to launch kernel");
  }

  clFinish(cmd_queue_);

  if (i % 2 == 0) {
    CopyDataBackFromDevice(dev_page_rank_);
  } else {
    CopyDataBackFromDevice(dev_page_rank_temp_);
  }
}

void PrCl20Benchmark::Cleanup() {
  PrBenchmark::Cleanup();

  cl_int ret;
  ret = clReleaseKernel(pr_kernel_);
  ret = clReleaseProgram(program_);

  clSVMFree(context_, dev_page_rank_);
  clSVMFree(context_, dev_page_rank_temp_);
  clSVMFree(context_, dev_row_offsets_);
  clSVMFree(context_, dev_column_numbers_);
  clSVMFree(context_, dev_values_);

  checkOpenCLErrors(ret, "Release objects.\n");
}
