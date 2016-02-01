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
  num_total_data_ = num_data_ * num_blocks;

  InitializeBuffer();
  InitializeData();

  InitCL();
  InitKernels();
}

void FIR::InitCL() {
  runtime_ = clRuntime::getInstance();

  platform_ = runtime_->getPlatformID();
  device_ = runtime_->getDevice();
  context_ = runtime_->getContext();
  cmd_queue_ = runtime_->getCmdQueue(0);

  file_ = clFile::getInstance();
}

void FIR::InitKernels() {
  cl_int err;
  file->open("fir_cl12_kernel.cl");

  const char *source = file->getSourceChar();
  program =
      clCreateProgramWithSource(context, 1, (const char **)&source, NULL, &err);
  checkOpenCLErrors(err, "Failed to create program with source...\n");

  err = clBuildProgram(program_, 0, NULL, NULL, NULL, NULL);
  checkOpenCLErrors(err, "Failed to create program...\n");

  kernel_fir_ = clCreateKernel(program_, "FIR". & err);
  checkOpenCLErrors(err, "Failed to create kernel FIR\n");
}

void FIR::InitializeBuffers() {
  input = (cl_float *)malloc(num_total_data_ * sizeof(cl_float));
  output = (cl_float *)malloc(num_total_data_ * sizeof(cl_float));
  coeff = (cl_float *)malloc(num_tap_ * sizeof(cl_float));
  temp_output = (cl_float *)malloc((num_data + num_tap - 1) * sizeof(cl_float));
}

void FIR::InitializeData() {
  for (unsigned i = 0; i < num_total_data; i++) {
    input_[i] = 8;
  }

  for (unsigned i = 0; i < num_tap; i++) {
    coeff_[i] = 1.0 / num_tap;
  }

  for (unsigned i = 0; i < (num_data + num_tap - 1); i++) {
    temp_output_[i] = 0.0;
  }
}

void FIR::Run() {
  // Create memory buffers on the device for each vector
  cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                      sizeof(cl_float) * num_data, NULL, &ret);
  cl_mem output_buffer = clCreateBuffer(
      context, CL_MEM_READ_WRITE, sizeof(cl_float) * num_data, NULL, &ret);
  cl_mem coeffBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                      sizeof(cl_float) * num_tap, NULL, &ret);
  cl_mem temp_output_buffer =
      clCreateBuffer(context, CL_MEM_READ_WRITE,
                     sizeof(cl_float) * (num_data + num_tap - 1), NULL, &ret);

  // Set the arguments of the kernel
  ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&output_buffer);
  ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&coeffBuffer);
  ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&temp_output_buffer);
  ret = clSetKernelArg(kernel, 3, sizeof(cl_uint), (void *)&num_tap);

  // Initialize Memory Buffer
  ret = clEnqueueWriteBuffer(command_queue, coeffBuffer, 1, 0,
                             num_tap * sizeof(cl_float), coeff, 0, 0, &event);

  event_list->add(event);

  ret = clEnqueueWriteBuffer(command_queue, temp_output_buffer, 1, 0,
                             (num_tap) * sizeof(cl_float), temp_output, 0, 0,
                             &event);

  event_list->add(event);

  // Decide the local group formation
  size_t globalThreads[1] = {num_data};
  size_t localThreads[1] = {128};
  // cl_command_type cmdType;
  count = 0;

  clock_gettime(CLOCK_MONOTONIC, &start); /* mark start time */

  while ((unsigned)count < num_blocks) {
    /* fill in the temp_input buffer object */
    ret = clEnqueueWriteBuffer(
        command_queue, temp_output_buffer, 1, (num_tap - 1) * sizeof(cl_float),
        num_data * sizeof(cl_float), input + (count * num_data), 0, 0, &event);

    // (num_tap-1)*sizeof(cl_float)
    event_list->add(event);

    // size_t global_item_size = num_data;
    // GLOBAL ITEMSIZE IS CUSTOM BASED ON COMPUTAION ALGO
    // size_t local_item_size = num_data;
    // size_t local_item_size[4] =
    // {num_data/4,num_data/4,num_data/4,num_data/4};
    // LOCAL ITEM SIZE IS CUSTOM BASED ON COMPUTATION ALGO

    // Execute the OpenCL kernel on the list
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, globalThreads,
                                 localThreads, 0, NULL, &event);

    CHECK_STATUS(ret, "Error: Range kernel. (clCreateKernel)\n");
    clFinish(command_queue);
    //    ret = clWaitForEvents(1, &event);
    //    ret = clWaitForEvents(1, &event);

    event_list->add(event);

    /* Get the output buffer */
    ret = clEnqueueReadBuffer(command_queue, output_buffer, CL_TRUE, 0,
                              num_data * sizeof(cl_float),
                              output + count * num_data, 0, NULL, &event);
    event_list->add(event);
    count++;
  }

  clock_gettime(CLOCK_MONOTONIC, &end); /* mark the end time */

  /* Uncomment to print output */
  // printf("\n The Output:\n");
  // i = 0;
  // while( i<num_total_data )
  // {
  //   printf( "%f ", output[i] );

  //   i++;
  // }

  ret = clFlush(command_queue);
  ret = clFinish(command_queue);
  ret = clReleaseKernel(kernel);
  ret = clReleaseProgram(program);
  ret = clReleaseMemObject(inputBuffer);
  ret = clReleaseMemObject(output_buffer);
  ret = clReleaseMemObject(coeffBuffer);
  ret = clReleaseMemObject(temp_output_buffer);
  ret = clReleaseCommandQueue(command_queue);
  ret = clReleaseContext(context);

  free(input);
  free(output);
  free(coeff);
  free(temp_output);

  /* Ensure that eventDumps exists */
  mkdir("eventDumps", 0700);

  /* comment to hide timing events */
  event_list->printEvents();
  event_list->dumpEvents((char *)"eventDumps");

  diff = BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
  printf("elapsed time = %llu nanoseconds\n", (long long unsigned int)diff);
}
