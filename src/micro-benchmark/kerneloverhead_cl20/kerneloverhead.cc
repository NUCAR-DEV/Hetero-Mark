/*
 * Hetero-Mark
 *
 * Copyright (c) 2015 Northeastern University
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
 * Microbenchmark to measure how long it takes to launch a kernel
 * in async way or sync way
 *
 * It takes a plain text or hex file and encrypts it with a given key
 *
 */


#define __NO_STD_VECTOR
#define MAX_SOURCE_SIZE (0x100000)
#include <stdio.h>/* for printf */
#include <stdint.h>/* for uint64 definition */
#include <stdlib.h>/* for exit() definition */
#include <time.h>/* for clock_gettime */
#include <string.h>

#include <CL/cl.h>
#include "include/kerneloverhead.h"

#define BILLION 1000000000L

void KernelOverhead::Run() {
  int maxnum = max_len;

  uint64_t diff;
  struct timespec start, end;

  FILE *cl_code = fopen("kernel.cl", "r");
  if (cl_code == NULL) { printf("\nerror: clfile\n"); exit(1); }
  char *source_str = (char *)malloc(MAX_SOURCE_SIZE);
  int res = fread(source_str, 1, MAX_SOURCE_SIZE, cl_code);
  fclose(cl_code);
  size_t source_length = strlen(source_str);

  cl_int err;
  cl_platform_id platform;
  cl_context context;
  cl_command_queue queue;
  cl_device_id device;
  cl_program program;

  err = clGetPlatformIDs(1, &platform, NULL);
  if (err != CL_SUCCESS) { printf("platformid %i", err); exit(1); }

  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
  if (err != CL_SUCCESS) { printf("deviceid %i", err); exit(1); }

  context = clCreateContext(0, 1, &device, NULL, NULL, &err);
  if (err != CL_SUCCESS) { printf("createcontext %i", err); exit(1); }

  queue = clCreateCommandQueueWithProperties(context, device, NULL, &err);
  if (err != CL_SUCCESS) { printf("commandqueue %i", err); exit(1); }

  program = clCreateProgramWithSource(context,
                                      1,
                                      (const char**)&source_str,
                                      &source_length, &err);

  if (err != CL_SUCCESS) {
    printf("createprogram %i", err); exit(1);
  }
  err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
  if (err != CL_SUCCESS) {
    printf("buildprogram ocl12 %i", err);
  }

  if (err == CL_BUILD_PROGRAM_FAILURE) {
    size_t log_size;
    clGetProgramBuildInfo(program,
                          device,
                          CL_PROGRAM_BUILD_LOG,
                          0,
                          NULL,
                          &log_size);

    char *log = (char *) malloc(log_size);
    clGetProgramBuildInfo(program,
                          device,
                          CL_PROGRAM_BUILD_LOG,
                          log_size,
                          log,
                          NULL);

    printf("%s\n", log);
    exit(1);
  }
  int i;
  cl_kernel kernel = clCreateKernel(program, "CLRunner", &err);
  if (err != CL_SUCCESS) { printf("createkernel %i", err); }
  cl_event event;
  const size_t local = 1;
  const size_t global = 1;

  err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global,
                               &local, 0, NULL, NULL);
  if (err != CL_SUCCESS) { printf("enqueuendrangekernel %i", err); }
  clFinish(queue);

  //  for (int xx = 1; xx < 11; xx++) {

  //  maxnum = xx * 100;

  printf("\nNumber of kernels = %i\n", maxnum);
  clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < maxnum; i++) {
      err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global,
                                   &local, 0, NULL, NULL);
      if (err != CL_SUCCESS) { printf("enqueuendrangekernel %i", err); }
      clFinish(queue);
    }

    clock_gettime(CLOCK_MONOTONIC, &end);

    diff = BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
    printf("\n\tTest-1: 'sync' done, time: %llu nanoseconds\n",
           (long long unsigned int) diff);

    // async
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < maxnum; i++) {
      err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global,
                                   &local, 0, NULL, NULL);
      if (err != CL_SUCCESS) { printf("enqueuendrangekernel %i", err); }
    }
    clFinish(queue);

    clock_gettime(CLOCK_MONOTONIC, &end);

    diff = BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
    printf("\n\tTest-2: 'async' done, time: %llu nanoseconds\n",
           (long long unsigned int) diff);

  clReleaseContext(context);
  clReleaseCommandQueue(queue);
  printf("\n");
}
