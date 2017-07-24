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

#include "src/fir/cl20/fir_cl20_benchmark.h"
#include <stdio.h>
#include <string.h>
#include <cstdlib>

void FirCl20Benchmark::Initialize() {
  FirBenchmark::Initialize();

  ClBenchmark::InitializeCl();
  InitializeKernels();

  InitializeBuffers();
  InitializeData();
}

void FirCl20Benchmark::InitializeKernels() {
  cl_int err;
  file_->open("kernels.cl");

  const char *source = file_->getSourceChar();
  program_ = clCreateProgramWithSource(context_, 1, (const char **)&source,
                                       NULL, &err);
  checkOpenCLErrors(err, "Failed to create program with source...\n");

  err =
      clBuildProgram(program_, 1, &device_, "-I ./ -cl-std=CL2.0", NULL, NULL);
  checkOpenCLErrors(err, "Failed to create program...\n");

  fir_kernel_ = clCreateKernel(program_, "FIR", &err);
  checkOpenCLErrors(err, "Failed to create kernel FIR\n");
}

void FirCl20Benchmark::InitializeBuffers() {
  input_svm_ = reinterpret_cast<cl_float *>(clSVMAlloc(
      context_, CL_MEM_READ_ONLY, num_data_per_block_ * sizeof(cl_float), 0));
  output_svm_ = reinterpret_cast<cl_float *>(clSVMAlloc(
      context_, CL_MEM_READ_WRITE, num_data_per_block_ * sizeof(cl_float), 0));
  history_ = reinterpret_cast<cl_float *>(
      clSVMAlloc(context_, CL_MEM_READ_WRITE, num_tap_ * sizeof(cl_float), 0));
  coeff_svm_ = reinterpret_cast<cl_float *>(
      clSVMAlloc(context_, CL_MEM_READ_WRITE, num_tap_ * sizeof(cl_float), 0));
}

void FirCl20Benchmark::InitializeData() {
  MapSvmBuffers();

  for (unsigned i = 0; i < num_tap_; i++) {
    coeff_svm_[i] = coeff_[i];
  }

  for (unsigned i = 0; i < num_tap_; i++) {
    history_[i] = 0.0;
  }

  UnmapSvmBuffers();
}

void FirCl20Benchmark::MapSvmBuffers() {
  cl_int err;

  err = clEnqueueSVMMap(cmd_queue_, CL_TRUE, CL_MAP_WRITE, input_svm_,
                        num_data_per_block_ * sizeof(cl_float), 0, 0, 0);
  checkOpenCLErrors(err, "Map SVM input\n");

  err = clEnqueueSVMMap(cmd_queue_, CL_TRUE, CL_MAP_WRITE, output_svm_,
                        num_data_per_block_ * sizeof(cl_float), 0, 0, 0);
  checkOpenCLErrors(err, "Map SVM output\n");

  err = clEnqueueSVMMap(cmd_queue_, CL_TRUE, CL_MAP_WRITE, coeff_svm_,
                        num_tap_ * sizeof(cl_float), 0, 0, 0);
  checkOpenCLErrors(err, "Map SVM coeff\n");

  err = clEnqueueSVMMap(cmd_queue_, CL_TRUE, CL_MAP_WRITE, history_,
                        num_tap_ * sizeof(cl_float), 0, 0, 0);
  checkOpenCLErrors(err, "Map SVM history\n");

  clFinish(cmd_queue_);
}

void FirCl20Benchmark::UnmapSvmBuffers() {
  cl_int err;

  err = clEnqueueSVMUnmap(cmd_queue_, input_svm_, 0, 0, 0);
  checkOpenCLErrors(err, "Ummap SVM input\n");

  err = clEnqueueSVMUnmap(cmd_queue_, output_svm_, 0, 0, 0);
  checkOpenCLErrors(err, "Ummap SVM output\n");

  err = clEnqueueSVMUnmap(cmd_queue_, coeff_svm_, 0, 0, 0);
  checkOpenCLErrors(err, "Ummap SVM coeff\n");

  err = clEnqueueSVMUnmap(cmd_queue_, history_, 0, 0, 0);
  checkOpenCLErrors(err, "Ummap SVM history\n");

  clFinish(cmd_queue_);
}

void FirCl20Benchmark::Run() {
  cl_int ret;

  // Set the arguments of the kernel
  ret = clSetKernelArgSVMPointer(fir_kernel_, 0, input_svm_);
  checkOpenCLErrors(ret, "Set kernel argument 0\n");

  ret = clSetKernelArgSVMPointer(fir_kernel_, 1, output_svm_);
  checkOpenCLErrors(ret, "Set kernel argument 1\n");

  ret = clSetKernelArgSVMPointer(fir_kernel_, 2, coeff_svm_);
  checkOpenCLErrors(ret, "Set kernel argument 2\n");

  ret = clSetKernelArgSVMPointer(fir_kernel_, 3, history_);
  checkOpenCLErrors(ret, "Set kernel argument 3\n");

  ret = clSetKernelArg(fir_kernel_, 4, sizeof(cl_uint),
                       reinterpret_cast<void *>(&num_tap_));
  checkOpenCLErrors(ret, "Set kernel argument 4\n");

  // Decide the local group formation
  unsigned int count = 0;
  size_t globalThreads[1] = {num_data_per_block_};
  size_t localThreads[1] = {64};

  MapSvmBuffers();
  while (count < num_block_) {
    memcpy(input_svm_, input_ + num_data_per_block_ * count,
           num_data_per_block_ * sizeof(float));
    UnmapSvmBuffers();

    // Execute the OpenCL kernel on the list
    ret = clEnqueueNDRangeKernel(cmd_queue_, fir_kernel_, CL_TRUE, NULL,
                                 globalThreads, localThreads, 0, NULL, NULL);
    checkOpenCLErrors(ret, "Enqueue ND Range.\n");
    clFinish(cmd_queue_);

    MapSvmBuffers();
    memcpy(output_ + num_data_per_block_ * count, output_svm_,
           num_data_per_block_ * sizeof(float));
    count++;
  }

  ret = clFlush(cmd_queue_);
  ret = clFinish(cmd_queue_);
  MapSvmBuffers();
}

void FirCl20Benchmark::Cleanup() {
  FirBenchmark::Cleanup();
  cl_int ret;
  ret = clReleaseKernel(fir_kernel_);
  ret = clReleaseProgram(program_);
  clSVMFree(context_, input_svm_);
  clSVMFree(context_, output_svm_);
  clSVMFree(context_, coeff_svm_);
  clSVMFree(context_, history_);
  checkOpenCLErrors(ret, "Release objects.\n");
}
