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

#include "src/fir/cl12/fir_cl12_benchmark.h"
#include <stdio.h>
#include <string.h>
#include <cstdlib>

void FirCl12Benchmark::Initialize() {
  FirBenchmark::Initialize();

  ClBenchmark::InitializeCl();
  InitializeKernels();

  InitializeBuffers();
  InitializeData();
}

void FirCl12Benchmark::InitializeKernels() {
  cl_int err;
  file_->open("kernels.cl");

  const char *source = file_->getSourceChar();
  program_ = clCreateProgramWithSource(context_, 1, (const char **)&source,
                                       NULL, &err);
  checkOpenCLErrors(err, "Failed to create program with source...\n");

  err = clBuildProgram(program_, 0, NULL, NULL, NULL, NULL);
  checkOpenCLErrors(err, "Failed to create program...\n");

  fir_kernel_ = clCreateKernel(program_, "FIR", &err);
  checkOpenCLErrors(err, "Failed to create kernel FIR\n");
}

void FirCl12Benchmark::InitializeBuffers() {
  // Create memory buffers on the device for each vector
  cl_int err;
  output_buffer_ =
      clCreateBuffer(context_, CL_MEM_READ_ONLY,
                     sizeof(cl_float) * num_data_per_block_, NULL, &err);
  checkOpenCLErrors(err, "Failed to allocate output buffer");

  coeff_buffer_ = clCreateBuffer(context_, CL_MEM_READ_ONLY,
                                 sizeof(cl_float) * num_tap_, NULL, &err);
  checkOpenCLErrors(err, "Failed to allocate coeff buffer");

  input_buffer_ =
      clCreateBuffer(context_, CL_MEM_READ_WRITE,
                     sizeof(cl_float) * num_data_per_block_, NULL, &err);
  checkOpenCLErrors(err, "Failed to allocate input buffer");

  history_buffer_ = clCreateBuffer(context_, CL_MEM_READ_ONLY,
                                   sizeof(cl_float) * num_tap_, NULL, &err);
  checkOpenCLErrors(err, "Failed to allocated history buffer");
}

void FirCl12Benchmark::InitializeData() {
  cl_int err;
  float init_value = 0;
  err = clEnqueueFillBuffer(cmd_queue_, history_buffer_,
                            static_cast<void *>(&init_value), sizeof(float), 0,
                            num_tap_ * sizeof(float), 0, NULL, NULL);
  checkOpenCLErrors(err, "Failed to fill in initial history buffer");
}

void FirCl12Benchmark::Run() {
  cl_int ret;
  unsigned int count;

  // Set the arguments of the kernel
  ret = clSetKernelArg(fir_kernel_, 0, sizeof(cl_mem),
                       static_cast<void *>(&output_buffer_));
  checkOpenCLErrors(ret, "Set kernel argument 0\n");
  ret = clSetKernelArg(fir_kernel_, 1, sizeof(cl_mem),
                       static_cast<void *>(&coeff_buffer_));
  checkOpenCLErrors(ret, "Set kernel argument 1\n");
  ret = clSetKernelArg(fir_kernel_, 2, sizeof(cl_mem),
                       static_cast<void *>(&input_buffer_));
  checkOpenCLErrors(ret, "Set kernel argument 2\n");
  ret = clSetKernelArg(fir_kernel_, 3, sizeof(cl_mem),
                       static_cast<void *>(&history_buffer_));
  checkOpenCLErrors(ret, "Set kernel argument 3\n");
  ret = clSetKernelArg(fir_kernel_, 4, sizeof(cl_uint),
                       static_cast<void *>(&num_tap_));
  checkOpenCLErrors(ret, "Set kernel argument 4\n");

  // Initialize Memory Buffer
  ret = clEnqueueWriteBuffer(cmd_queue_, coeff_buffer_, CL_TRUE, 0,
                             num_tap_ * sizeof(float), coeff_, 0, NULL, NULL);
  checkOpenCLErrors(ret, "Copy coeff to buffer\n");

  float init_value = 0;
  ret = clEnqueueFillBuffer(cmd_queue_, history_buffer_,
                            static_cast<void *>(&init_value), sizeof(float), 0,
                            num_tap_ * sizeof(float), 0, NULL, NULL);
  checkOpenCLErrors(ret, "Set history buffer\n");

  ret = clEnqueueFillBuffer(cmd_queue_, input_buffer_,
                            static_cast<void *>(&init_value), sizeof(float), 0,
                            num_data_per_block_ * sizeof(float), 0, NULL, NULL);
  checkOpenCLErrors(ret, "Set input buffer\n");

  // Decide the local group formation
  size_t globalThreads[1] = {num_data_per_block_};
  size_t localThreads[1] = {64};
  count = 0;

  while (count < num_block_) {
    // Fill in the history buffer
    if (count > 0) {
      ret = clEnqueueWriteBuffer(
          cmd_queue_, history_buffer_, CL_TRUE, 0, num_tap_ * sizeof(cl_float),
          input_ + (count * num_data_per_block_) - num_tap_, 0, 0, NULL);
      checkOpenCLErrors(ret, "Copy history to buffer\n");
    }

    // Fill in the input buffer object
    ret = clEnqueueWriteBuffer(cmd_queue_, input_buffer_, CL_TRUE, 0,
                               num_data_per_block_ * sizeof(cl_float),
                               input_ + (count * num_data_per_block_), 0, 0,
                               NULL);
    checkOpenCLErrors(ret, "Copy data to buffer\n");

    cpu_gpu_logger_->GPUOn();
    // Execute the OpenCL kernel on the list
    ret = clEnqueueNDRangeKernel(cmd_queue_, fir_kernel_, CL_TRUE, NULL,
                                 globalThreads, localThreads, 0, NULL, NULL);
    checkOpenCLErrors(ret, "Enqueue ND Range.\n");
    clFinish(cmd_queue_);
    cpu_gpu_logger_->GPUOff();

    // Get the output buffer
    ret = clEnqueueReadBuffer(cmd_queue_, output_buffer_, CL_TRUE, 0,
                              num_data_per_block_ * sizeof(cl_float),
                              output_ + count * num_data_per_block_, 0, NULL,
                              NULL);
    checkOpenCLErrors(ret, "Copy data back\n");
    clFinish(cmd_queue_);

    count++;
  }

  ret = clFlush(cmd_queue_);
  ret = clFinish(cmd_queue_);
  cpu_gpu_logger_->Summarize();
}

void FirCl12Benchmark::Cleanup() {
  FirBenchmark::Cleanup();

  cl_int ret;
  ret = clReleaseKernel(fir_kernel_);
  ret |= clReleaseProgram(program_);
  ret |= clReleaseMemObject(output_buffer_);
  ret |= clReleaseMemObject(coeff_buffer_);
  ret |= clReleaseMemObject(input_buffer_);
  ret |= clReleaseMemObject(history_buffer_);
  checkOpenCLErrors(ret, "Release objects.\n");
}
