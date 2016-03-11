/* Copyright (c) 2015 Northeastern University
 * All rights reserved.
 *
 * Developed by:Northeastern University Computer Architecture Research (NUCAR)
 * Group, Northeastern University, http://www.ece.neu.edu/groups/nucar/
 *
 * Author: Carter McCardwell (carter@mccardwell.net, cmccardw@ece.neu.edu)
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

#include "src/aes/cl12/aes_cl12_benchmark.h"
#include <string>
#include <memory>

void AesCl12Benchmark::InitializeKernel() {
  cl_int err;

  // Create program
  file_->open("kernels.cl");
  const char *source = file_->getSourceChar();
  program_ = clCreateProgramWithSource(context_, 1, (const char **)&source,
                                       NULL, &err);
  if (err != CL_SUCCESS) {
    char buf[0x10000];
    clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, 0x10000, buf,
                          NULL);
    printf("Build info:\n%s\n", buf);
    exit(-1);
  }

  err = clBuildProgram(program_, 1, &device_, "-I ./ -cl-std=CL1.2", NULL, NULL);
  checkOpenCLErrors(err, "Failed to build program...\n");

  kernel_ = clCreateKernel(program_, "Encrypt", &err);
  checkOpenCLErrors(err, "Failed to create AES kernel\n");
}

void AesCl12Benchmark::Initialize() {
  AesBenchmark::Initialize();
  ClBenchmark::InitializeCl();
  InitializeKernel();
  InitializeDeviceMemory();
}

void AesCl12Benchmark::InitializeDeviceMemory() {
  cl_int err;
  dev_ciphertext_ = clCreateBuffer(context_, CL_MEM_READ_WRITE,
                                   text_length_, NULL, &err);
  checkOpenCLErrors(err, "Failed to create buffer for ciphertext");

  dev_key_ = clCreateBuffer(context_, CL_MEM_READ_ONLY,
                            kExpandedKeyLengthInBytes, NULL, &err);
  checkOpenCLErrors(err, "Failed to create buffer for expanded key");
}

void AesCl12Benchmark::Cleanup() {
  AesBenchmark::Cleanup();
  FreeKernel();
  FreeDeviceMemory();
}

void AesCl12Benchmark::FreeKernel() {
  cl_int err;
  err = clReleaseKernel(kernel_);
  err = clReleaseProgram(program_);
  checkOpenCLErrors(err, "Failed to release kernel");
}

void AesCl12Benchmark::FreeDeviceMemory() {
  cl_int ret;
  ret = clReleaseMemObject(dev_ciphertext_);
  ret = clReleaseMemObject(dev_key_);
  checkOpenCLErrors(ret, "ReleaseDeviceBuffers");
}

void AesCl12Benchmark::Run() {
  ExpandKey();
  CopyDataToDevice();
  RunKernel();
  CopyDataBackFromDevice();
}

void AesCl12Benchmark::CopyDataToDevice() {
  cl_int ret;
  ret = clEnqueueWriteBuffer(cmd_queue_, dev_ciphertext_, CL_TRUE, 0,
                             text_length_, ciphertext_, 0, NULL, NULL);
  checkOpenCLErrors(ret, "Failed to copy cipher text to device");

  ret = clEnqueueWriteBuffer(cmd_queue_, dev_key_, CL_TRUE, 0,
                             kExpandedKeyLengthInBytes, expanded_key_, 0, NULL,
                             NULL);
  checkOpenCLErrors(ret, "Failed to copy key to device");
}

void AesCl12Benchmark::RunKernel() {
  cl_int ret;

  int num_blocks = text_length_ / 16;
  size_t global_dimensions[] = {static_cast<size_t>(num_blocks)};
  size_t local_dimensions[] = {64};

  ret = clSetKernelArg(kernel_, 0, sizeof(cl_mem), &dev_ciphertext_);
  checkOpenCLErrors(ret, "Set ciphertext as kernel argument");

  ret = clSetKernelArg(kernel_, 1, sizeof(cl_mem), &dev_key_);
  checkOpenCLErrors(ret, "Set key as kernel argument");

  ret = clEnqueueNDRangeKernel(cmd_queue_, kernel_, 1, NULL, global_dimensions,
                               local_dimensions, 0, NULL, NULL);
  checkOpenCLErrors(ret, "Launch kernel");

  clFinish(cmd_queue_);
}

void AesCl12Benchmark::CopyDataBackFromDevice() {
  cl_int ret;

  ret = clEnqueueReadBuffer(cmd_queue_, dev_ciphertext_, CL_TRUE, 0,
                            text_length_, ciphertext_, 0, NULL, NULL);
  checkOpenCLErrors(ret, "Read buffer from device");
}
