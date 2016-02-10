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

#include <string.h>
#include <stdio.h>
#include <cstdlib>
#include <sys/stat.h>

#include "src/opencl20/fir_cl20/fir_cl20.h"

void FIR::Initialize() {
  num_total_data_ = num_data_ * num_blocks_;

  InitializeCL();
  InitializeKernels();

  InitializeBuffers();
  InitializeData();
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
  timer->End({"Initialize"});
  timer->Start();
  file_->open("fir_cl20_kernel.cl");

  const char *source = file_->getSourceChar();
  program_ = clCreateProgramWithSource(context_, 1, (const char **)&source,
                                       NULL, &err);
  checkOpenCLErrors(err, "Failed to create program with source...\n");

  err =
      clBuildProgram(program_, 1, &device_, "-I ./ -cl-std=CL2.0", NULL, NULL);
  checkOpenCLErrors(err, "Failed to build program...\n");

  fir_kernel_ = clCreateKernel(program_, "FIR", &err);
  checkOpenCLErrors(err, "Failed to create kernel FIR\n");
  timer->End({"Compilation"});
  timer->Start();
}

void FIR::InitializeBuffers() {
  input_ = (cl_float *)clSVMAlloc(context_, CL_MEM_READ_ONLY,
                                  num_total_data_ * sizeof(cl_float), 0);
  output_ = (cl_float *)clSVMAlloc(context_, CL_MEM_READ_WRITE,
                                   num_total_data_ * sizeof(cl_float), 0);
  coeff_ = (cl_float *)clSVMAlloc(context_, CL_MEM_READ_ONLY,
                                  num_tap_ * sizeof(cl_float), 0);
  history_ = (cl_float *)clSVMAlloc(context_, CL_MEM_READ_WRITE,
                                    num_tap_ * sizeof(cl_float), 0);
}

void FIR::InitializeData() {
  srand(time(NULL));

  MapSvmBuffers();

  for (unsigned i = 0; i < num_total_data_; i++) {
    input_[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  }

  for (unsigned i = 0; i < num_tap_; i++) {
    coeff_[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  }

  for (unsigned i = 0; i < num_tap_; i++) {
    history_[i] = 0.0;
  }

  UnmapSvmBuffers();
}

void FIR::MapSvmBuffers() {
  cl_int err;

  err = clEnqueueSVMMap(cmd_queue_, CL_TRUE, CL_MAP_WRITE, input_,
                        num_total_data_ * sizeof(cl_float), 0, 0, 0);
  checkOpenCLErrors(err, "Map SVM input\n");

  err = clEnqueueSVMMap(cmd_queue_, CL_TRUE, CL_MAP_WRITE, output_,
                        num_total_data_ * sizeof(cl_float), 0, 0, 0);
  checkOpenCLErrors(err, "Map SVM output\n");

  err = clEnqueueSVMMap(cmd_queue_, CL_TRUE, CL_MAP_WRITE, coeff_,
                        num_tap_ * sizeof(cl_float), 0, 0, 0);
  checkOpenCLErrors(err, "Map SVM coeff\n");

  err = clEnqueueSVMMap(cmd_queue_, CL_TRUE, CL_MAP_WRITE, history_,
                        num_tap_ * sizeof(cl_float), 0, 0, 0);
  checkOpenCLErrors(err, "Map SVM history\n");
}

void FIR::UnmapSvmBuffers() {
  cl_int err;

  err = clEnqueueSVMUnmap(cmd_queue_, input_, 0, 0, 0);
  checkOpenCLErrors(err, "Ummap SVM input\n");

  err = clEnqueueSVMUnmap(cmd_queue_, output_, 0, 0, 0);
  checkOpenCLErrors(err, "Ummap SVM output\n");

  err = clEnqueueSVMUnmap(cmd_queue_, coeff_, 0, 0, 0);
  checkOpenCLErrors(err, "Ummap SVM input\n");

  err = clEnqueueSVMUnmap(cmd_queue_, history_, 0, 0, 0);
  checkOpenCLErrors(err, "Ummap SVM history\n");
}

void FIR::Run() {
  cl_int ret;
  unsigned int count;

  // Set the arguments of the kernel
  ret = clSetKernelArgSVMPointer(fir_kernel_, 2, coeff_);
  checkOpenCLErrors(ret, "Set kernel argument 2\n");

  ret = clSetKernelArgSVMPointer(fir_kernel_, 3, history_);
  checkOpenCLErrors(ret, "Set kernel argument 3\n");

  ret = clSetKernelArg(fir_kernel_, 4, sizeof(cl_uint), (void *)&num_tap_);
  checkOpenCLErrors(ret, "Set kernel argument 4\n");

  // Decide the local group formation
  size_t globalThreads[1] = {num_data_};
  size_t localThreads[1] = {64};
  count = 0;

  while (count < num_blocks_) {
    ret = clSetKernelArgSVMPointer(fir_kernel_, 0, input_ + num_data_ * count);
    checkOpenCLErrors(ret, "Set kernel argument 0\n");

    ret = clSetKernelArgSVMPointer(fir_kernel_, 1, output_ + num_data_ * count);
    checkOpenCLErrors(ret, "Set kernel argument 1\n");

    // Execute the OpenCL kernel on the list
    ret = clEnqueueNDRangeKernel(cmd_queue_, fir_kernel_, CL_TRUE, NULL,
                                 globalThreads, localThreads, 0, NULL, NULL);
    checkOpenCLErrors(ret, "Enqueue ND Range.\n");
    clFinish(cmd_queue_);

    count++;
  }

  ret = clFlush(cmd_queue_);
  ret = clFinish(cmd_queue_);
  MapSvmBuffers();
}

void FIR::Verify() {
  bool has_error = false;
  float *cpu_output = new float[num_total_data_];
  for (unsigned int i = 0; i < num_total_data_; i++) {
    float sum = 0;
    for (unsigned int j = 0; j < num_tap_; j++) {
      if (i < j) continue;
      sum += input_[i - j] * coeff_[j];
    }
    cpu_output[i] = sum;
    if (abs(cpu_output[i] - output_[i]) > 1e-5) {
      has_error = true;
      printf("At position %d, expected %f, but was %f.\n", i, cpu_output[i],
             output_[i]);
    }
  }

  if (!has_error) {
    printf("Passed!\n");
  }

  delete cpu_output;
}

void FIR::Cleanup() {
  cl_int ret;
  ret = clReleaseKernel(fir_kernel_);
  ret = clReleaseProgram(program_);
  clSVMFree(context_, input_);
  clSVMFree(context_, output_);
  clSVMFree(context_, coeff_);
  clSVMFree(context_, history_);
  checkOpenCLErrors(ret, "Release objects.\n");
}
