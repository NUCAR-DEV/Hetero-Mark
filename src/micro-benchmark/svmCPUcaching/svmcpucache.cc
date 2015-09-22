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
 * To test if caching mechanism exists for Shared Visrtual Memory
 * (SVM) in CPU side
 *
 * It takes a plain text or hex file and encrypts it with a given key
 *
 */

#define __NO_STD_VECTOR
#define MAX_SOURCE_SIZE (0x100000)

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>

#include <CL/cl.h>
#include "include/svmcpucache.h"

#define BILLION 1000000000L

void CPUCache::Run() {
  uint64_t diff1, diff2;
  struct timespec start, end1, end2;

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

  program = clCreateProgramWithSource(context, 1, (const char**)&source_str,
                                      &source_length, &err);
  if (err != CL_SUCCESS) { printf("createprogram %i", err); exit(1); }

  err = clBuildProgram(program, 1, &device, "-I ./ -cl-std=CL2.0", NULL, NULL);
  if (err != CL_SUCCESS) { printf("buildprogram ocl20 %i", err); }

  if (err == CL_BUILD_PROGRAM_FAILURE) {
    size_t log_size;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                          0, NULL, &log_size);
    char *log = (char *) malloc(log_size);
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                          log_size, log, NULL);
    printf("%s\n", log);
    exit(1);
  }

  int i, j = 0;
  int sz = 1000000;
  int cline = 4;

  int *indata  = (int *)clSVMAlloc(context, CL_MEM_READ_WRITE,
                                   sizeof(int)*sz, 0);
  int *indata2  = (int *)clSVMAlloc(context, CL_MEM_READ_WRITE,
                                    sizeof(int)*sz, 0);
  int *out1 = (int *)malloc(sizeof(int)*sz);
  int *out2 = (int *)malloc(sizeof(int)*sz);
  int *out3 = (int *)malloc(sizeof(int)*sz);

  err = clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_WRITE,
                        indata, sizeof(int)*sz, 0, 0, 0);
  err = clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_WRITE,
                        indata2, sizeof(int)*sz, 0, 0, 0);

  if (err != CL_SUCCESS) { printf("enqueuesvmmap ocl20 %i", err); }

  for (i = 0; i < sz; i++) {   indata[i] = rand(); }

  clock_gettime(CLOCK_MONOTONIC, &start);/* mark start time */

  for (i = 0; i < sz; i = i+cline) {
    memcpy(&out1[i], &indata[i], sizeof(int)*cline);
  }

  clock_gettime(CLOCK_MONOTONIC, &end1);/* mark the end time */

  for (i = 0; i < sz; i = i+cline) {
    memcpy(&out2[i], &indata2[i], sizeof(int)*cline);
    memcpy(&out3[i], &indata2[i], sizeof(int)*cline);
  }

  clock_gettime(CLOCK_MONOTONIC, &end2);/* mark the end time */

  err = clEnqueueSVMUnmap(queue, indata, 0, 0, 0);

  if (err != CL_SUCCESS) { printf("enqueueunmap ocl20 %i", err); }

  err = clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_READ,
                        indata, sizeof(int)*sz, 0, 0, 0);

  err = clEnqueueSVMUnmap(queue, indata, 0, 0, 0);
  err = clFinish(queue);
  clSVMFree(context, indata);
  clSVMFree(context, indata2);
  free(out1);
  free(out2);
  free(out3);

  diff1 = BILLION * (end1.tv_sec - start.tv_sec) + end1.tv_nsec - start.tv_nsec;
  printf("\n Test 1 done, time: %llu nanoseconds\n",
         (long long unsigned int) diff1);

  diff2 = BILLION * (end2.tv_sec - end1.tv_sec) + end2.tv_nsec - end1.tv_nsec;
  printf("\n Test 2 done, time: %llu nanoseconds\n",
         (long long unsigned int) diff2);

  // printf("\n\n--Done--\n");
}

