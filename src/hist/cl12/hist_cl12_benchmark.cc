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

#include "src/hist/cl12/hist_cl12_benchmark.h"
#include <stdio.h>
#include <string.h>
#include <cstdlib>

void HistCl12Benchmark::Initialize() {
  HistBenchmark::Initialize();

  ClBenchmark::InitializeCl();

  InitializeKernels();
  InitializeBuffers();
}

void HistCl12Benchmark::InitializeKernels() {
  cl_int err;
  file_->open("kernels.cl");

  const char *source = file_->getSourceChar();
  program_ = clCreateProgramWithSource(context_, 1, (const char **)&source,
                                       NULL, &err);
  checkOpenCLErrors(err, "Failed to create program with source...\n");

  err = clBuildProgram(program_, 1, &device_, NULL, NULL, NULL);
  if (err != CL_SUCCESS) {
    char buf[0x10000];
    clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, 0x10000, buf,
                          NULL);
    printf("Build info:\n%s\n", buf);
    exit(-1);
  }
  checkOpenCLErrors(err, "Failed to create program...\n");

  hist_kernel_ = clCreateKernel(program_, "HIST", &err);
  checkOpenCLErrors(err, "Failed to create kernel HIST\n");
}

void HistCl12Benchmark::InitializeBuffers() {
  cl_int err;
  dev_pixels_ = clCreateBuffer(context_, CL_MEM_READ_ONLY,
                               num_pixel_ * sizeof(uint32_t), NULL, &err);
  checkOpenCLErrors(err, "Failed to create pixels buffer");

  dev_histogram_ = clCreateBuffer(context_, CL_MEM_READ_WRITE,
                                  num_color_ * sizeof(uint32_t), NULL, &err);
  checkOpenCLErrors(err, "Failed to create histogram buffer");

  uint32_t pattern[] = {0};
  err = clEnqueueFillBuffer(cmd_queue_, dev_histogram_, pattern, 1, 0,
                            num_color_ * sizeof(uint32_t), 0, NULL, NULL);
  checkOpenCLErrors(err, "Failed to initialize histogram buffer");
}

void HistCl12Benchmark::Run() {
  cl_int err;

  uint32_t pattern[] = {0};
  err = clEnqueueFillBuffer(cmd_queue_, dev_histogram_, pattern, 1, 0,
                            num_color_ * sizeof(uint32_t), 0, NULL, NULL);
  checkOpenCLErrors(err, "Failed to initialize histogram buffer");

  err = clEnqueueWriteBuffer(cmd_queue_, dev_pixels_, CL_TRUE, 0,
                             num_pixel_ * sizeof(uint32_t), pixels_, 0, NULL,
                             NULL);
  checkOpenCLErrors(err, "Failed to write pixels buffer");
  clFinish(cmd_queue_);

  err = clSetKernelArg(hist_kernel_, 0, sizeof(cl_mem), &dev_pixels_);
  checkOpenCLErrors(err, "Failed to set argument 0");
  err = clSetKernelArg(hist_kernel_, 1, sizeof(cl_mem), &dev_histogram_);
  checkOpenCLErrors(err, "Failed to set argument 1");
  err = clSetKernelArg(hist_kernel_, 2, sizeof(uint32_t), &num_color_);
  checkOpenCLErrors(err, "Failed to set argument 2");
  err = clSetKernelArg(hist_kernel_, 3, sizeof(uint32_t), &num_pixel_);
  checkOpenCLErrors(err, "Failed to set argument 3");

  size_t global_dimensions[] = {1024};
  size_t local_dimensions[] = {64};

  cpu_gpu_logger_->GPUOn();
  err = clEnqueueNDRangeKernel(cmd_queue_, hist_kernel_, CL_TRUE, NULL,
                               global_dimensions, local_dimensions, 0, 0, NULL);
  checkOpenCLErrors(err, "Failed to launch kernel");
  clFinish(cmd_queue_);
  cpu_gpu_logger_->GPUOff();

  err = clEnqueueReadBuffer(cmd_queue_, dev_histogram_, CL_TRUE, 0,
                            num_color_ * sizeof(uint32_t), histogram_, 0, NULL,
                            NULL);
  checkOpenCLErrors(err, "Failed to copy histogram back");
  clFinish(cmd_queue_);
  cpu_gpu_logger_->Summarize();
}

void HistCl12Benchmark::Cleanup() {
  HistBenchmark::Cleanup();

  cl_int ret;
  ret = clReleaseKernel(hist_kernel_);
  ret = clReleaseProgram(program_);
  ret = clReleaseMemObject(dev_pixels_);
  ret = clReleaseMemObject(dev_histogram_);
  checkOpenCLErrors(ret, "Release objects.\n");
}
