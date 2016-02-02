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
 *
 */
#include <string.h>
#include <time.h>/* for clock_gettime */
#include <stdio.h>
#include <stdint.h>/* for uint64 definition */
#include <stdlib.h>/* for exit() definition */
#include <CL/cl.h>
#include <sys/stat.h>

#include "src/opencl12/fir_cl12/fir_cl12.h"

#define CHECK_STATUS(status, message) \
  if (status != CL_SUCCESS) {         \
    printf(message);                  \
    printf("\n");                     \
  }

void FIR::Initialize() {
  num_total_data_ = num_data_ * num_blocks_;

  InitializeBuffers();
  InitializeData();

  InitializeCL();
  InitializeKernels();
}

void FIR::InitializeCL() {
  runtime_ = clRuntime::getInstance();

  platform_ = runtime_->getPlatformID();
  device_ = runtime_->getDevice();
  context_ = runtime_->getContext();
  cmd_queue_ = runtime_->getCmdQueue(0);

  file_ = clFile::getInstance();
}

void FIR::InitializeKernels() {
  cl_int err;
  file_->open("fir_cl12_kernel.cl");

  const char *source = file_->getSourceChar();
  program_ = clCreateProgramWithSource(context_, 1, 
                                      (const char **)&source, 
                                      NULL, &err);
  checkOpenCLErrors(err, "Failed to create program with source...\n");

  err = clBuildProgram(program_, 0, NULL, NULL, NULL, NULL);
  checkOpenCLErrors(err, "Failed to create program...\n");

  fir_kernel_ = clCreateKernel(program_, "FIR", &err);
  checkOpenCLErrors(err, "Failed to create kernel FIR\n");
}

void FIR::InitializeBuffers() {
  input_ = (cl_float *)malloc(num_total_data_ * sizeof(cl_float));
  output_ = (cl_float *)malloc(num_total_data_ * sizeof(cl_float));
  coeff_ = (cl_float *)malloc(num_tap_ * sizeof(cl_float));
  temp_output_ = (cl_float *)malloc(
      (num_data_ + num_tap_ - 1) * sizeof(cl_float));

  // Create memory buffers on the device for each vector
  input_buffer_ = clCreateBuffer(context_, CL_MEM_READ_ONLY,
                                 sizeof(cl_float) * num_data_, 
                                 NULL, NULL);
  output_buffer_ = clCreateBuffer(context_, CL_MEM_READ_WRITE, 
                                  sizeof(cl_float) * num_data_, 
                                  NULL, NULL);
  coeff_buffer_ = clCreateBuffer(context_, CL_MEM_READ_ONLY,
                                 sizeof(cl_float) * num_tap_, 
                                 NULL, NULL);
  temp_output_buffer_ = clCreateBuffer(
      context_, CL_MEM_READ_WRITE,
      sizeof(cl_float) * (num_data_ + num_tap_ - 1), NULL, NULL);
}

void FIR::InitializeData() {
  for (unsigned i = 0; i < num_total_data_; i++) {
    input_[i] = 8;
  }

  for (unsigned i = 0; i < num_tap_; i++) {
    coeff_[i] = 1.0 / num_tap_;
  }

  for (unsigned i = 0; i < (num_data_ + num_tap_ - 1); i++) {
    temp_output_[i] = 0.0;
  }
}

void FIR::Run() {
  cl_int ret;
  unsigned int count;

  // Set the arguments of the kernel
  ret = clSetKernelArg(fir_kernel_, 0, sizeof(cl_mem), 
                       (void *)&output_buffer_);
  ret = clSetKernelArg(fir_kernel_, 1, sizeof(cl_mem), 
                       (void *)&coeff_buffer_);
  ret = clSetKernelArg(fir_kernel_, 2, sizeof(cl_mem), 
                       (void *)&temp_output_buffer_);
  ret = clSetKernelArg(fir_kernel_, 3, sizeof(cl_uint), 
                       (void *)&num_tap_);

  // Initialize Memory Buffer
  ret = clEnqueueWriteBuffer(cmd_queue_, coeff_buffer_, 1, 0,
                             num_tap_ * sizeof(cl_float), 
                             coeff_, 0, 0, NULL);

  ret = clEnqueueWriteBuffer(cmd_queue_, temp_output_buffer_, 1, 0,
                             (num_tap_) * sizeof(cl_float), temp_output_, 0, 0,
                             NULL);

  // Decide the local group formation
  size_t globalThreads[1] = {num_data_};
  size_t localThreads[1] = {128};
  count = 0;

  while (count < num_blocks_) {
    // fill in the temp_input buffer object
    ret = clEnqueueWriteBuffer(
        cmd_queue_, temp_output_buffer_, 1, 
        ((num_tap_) - 1) * sizeof(cl_float),
        num_data_ * sizeof(cl_float), 
        input_ + (count * num_data_), 
        0, 0, NULL);

    // Execute the OpenCL kernel on the list
    ret = clEnqueueNDRangeKernel(cmd_queue_, fir_kernel_, 1, NULL, 
                                 globalThreads, localThreads, 0, NULL, NULL);
    CHECK_STATUS(ret, "Error: Range kernel. (clCreateKernel)\n");
    clFinish(cmd_queue_);

    // Get the output buffer
    ret = clEnqueueReadBuffer(cmd_queue_, output_buffer_, CL_TRUE, 0,
                              num_data_ * sizeof(cl_float),
                              output_ + count * num_data_, 
                              0, NULL, NULL);
    count++;
  }

  ret = clFlush(cmd_queue_);
  ret = clFinish(cmd_queue_);
}

void FIR::Cleanup() {
  cl_int ret;
  ret = clReleaseKernel(fir_kernel_);
  ret = clReleaseProgram(program_);
  ret = clReleaseMemObject(input_buffer_);
  ret = clReleaseMemObject(output_buffer_);
  ret = clReleaseMemObject(coeff_buffer_);
  ret = clReleaseMemObject(temp_output_buffer_);

  CHECK_STATUS(ret, "Error: Release objects.\n");

  free(input_);
  free(output_);
  free(coeff_);
  free(temp_output_);
}
