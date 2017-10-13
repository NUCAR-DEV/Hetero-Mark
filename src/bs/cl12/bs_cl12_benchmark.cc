/*
 * Hetero-mark
 *
 * Copyright (c) 2015 Northeastern University
 * All rights reserved.
 *
 * Developed by:
 *   Northeastern University Computer Architecture Research (NUCAR) Group
 *   Northeastern University
 *   http://www.ece.neu.edu/groups/nucar/
 *
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal with the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 *   Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimers.
 *
 *   Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimers in the
 *   documentation and/or other materials provided with the distribution.
 *
 *   Neither the names of NUCAR, Northeastern University, nor the names of
 *   its contributors may be used to endorse or promote products derived
 *   from this Software without specific prior written permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS WITH THE SOFTWARE.
 */

#include "src/bs/cl12/bs_cl12_benchmark.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

void BsCl12Benchmark::Initialize() {
  BsBenchmark::Initialize();

  ClBenchmark::InitializeCl();
  InitializeKernels();

  InitializeBuffers();
}

void BsCl12Benchmark::InitializeKernels() {
  cl_int err;
  file_->open("kernels.cl");

  const char *source = file_->getSourceChar();
  program_ = clCreateProgramWithSource(context_, 1, (const char **)&source,
				       NULL, &err);
  checkOpenCLErrors(err, "Failed to create program with source...\n");

  err = clBuildProgram(program_, 0, NULL, NULL, NULL, NULL);

  if(err!=CL_SUCCESS){
    size_t len;
    char *msg;
    // get the details on the error, and store it in buffer
    clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG,0,NULL,&len);
    msg=new char[len];
    clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG,len,msg,NULL);
    printf("Kernel build error:\n%s\n", msg);
    delete msg;
  }
  checkOpenCLErrors(err, "Failed to create program...\n");

  bs_kernel_ = clCreateKernel(program_, "bs_cl12", &err);
  checkOpenCLErrors(err, "Failed to create kernel BS\n");
}


void BsCl12Benchmark::InitializeBuffers() {
  BsBenchmark::Initialize();

  cl_int err;
  d_rand_array_ =
    clCreateBuffer(context_, CL_MEM_READ_WRITE,
		   num_tiles_ * tile_size_ * sizeof(float), NULL, &err);
  checkOpenCLErrors(err, "Failed to allocate rand array buffer");

  d_call_price_ =
    clCreateBuffer(context_, CL_MEM_READ_WRITE,
		   num_tiles_ * tile_size_ * sizeof(float), NULL, &err);
  checkOpenCLErrors(err, "Failed to allocate call price buffer");

  d_put_price_ =
    clCreateBuffer(context_, CL_MEM_READ_WRITE,
		   num_tiles_ * tile_size_ * sizeof(float), NULL, &err);
  checkOpenCLErrors(err, "Failed to allocate put price buffer");

  err = clEnqueueWriteBuffer(cmd_queue_, d_rand_array_, CL_TRUE, 0,
			     num_tiles_ * tile_size_ * sizeof(float), rand_array_, 0, NULL,
			     NULL);
  checkOpenCLErrors(err, "Copy rand array data\n");
  
  float *temp_call_price = reinterpret_cast<float *>(
      malloc(num_tiles_ * tile_size_ * sizeof(float)));
  float *temp_put_price = reinterpret_cast<float *>(
      malloc(num_tiles_ * tile_size_ * sizeof(float)));
  for (unsigned int i = 0; i < num_tiles_ * tile_size_; i++) {
    temp_call_price[i] = 0;
    temp_put_price[i] = 0;
  }

  err = clEnqueueWriteBuffer(cmd_queue_, d_call_price_, CL_TRUE, 0,
			     num_tiles_ * tile_size_ * sizeof(float), temp_call_price, 0, NULL,
			     NULL);
  checkOpenCLErrors(err, "Copy call price data\n");

  err = clEnqueueWriteBuffer(cmd_queue_, d_put_price_, CL_TRUE, 0,
			     num_tiles_ * tile_size_ * sizeof(float), temp_put_price, 0, NULL,
			     NULL);
  checkOpenCLErrors(err, "Copy rand put price data\n");
  
  free(temp_call_price);
  free(temp_put_price);

  event_ = clCreateUserEvent(context_, &err);
  checkOpenCLErrors(err, "Create event\n");

  // Set to completed to allow first GPU kernel to launch
  err = clSetUserEventStatus(event_, CL_COMPLETE);
  checkOpenCLErrors(err, "Set event status\n");
}

void BsCl12Benchmark::Run() {
  // The main while loop
  uint32_t done_tiles_ = 0;
  uint32_t last_tile_ = num_tiles_;

  cl_int err;
  
  // while the done tiles are less than num_tiles, continue
  while (done_tiles_ < last_tile_) {
    // First check to make sure that we are launching the first set
    if (IsGpuCompleted()) {
      // No longer the first lunch after this point so
      // turn it off
      // printf("Completion set to 1. GPU running \n");

      // Set the size of the section based on the number of tiles
      // and the number of compute units
      uint32_t section_tiles = (gpu_chunk_ < last_tile_ - done_tiles_)
                                   ? gpu_chunk_
                                   : last_tile_ - done_tiles_;

      unsigned int offset = done_tiles_ * tile_size_;
      //               printf("Section tile is %d \n", section_tiles);

      // GPU is running the following tiles
      // fprintf(stderr, "GPU tiles: %d to %d\n", done_tiles_,
      //       done_tiles_ + section_tiles);
      done_tiles_ += section_tiles;

      size_t localThreads[1] = {64};
      size_t globalThreads[1] = {(section_tiles * tile_size_)};

      // Set the arguments of the kernel
      err = clSetKernelArg(bs_kernel_, 0, sizeof(cl_mem),
			   reinterpret_cast<void *>(&d_rand_array_));
      checkOpenCLErrors(err, "Set kernel argument 0\n");

      err = clSetKernelArg(bs_kernel_, 1, sizeof(cl_mem),
			   reinterpret_cast<void *>(&d_call_price_));
      checkOpenCLErrors(err, "Set kernel argument 1\n");

      err = clSetKernelArg(bs_kernel_, 2, sizeof(cl_mem),
			   reinterpret_cast<void *>(&d_put_price_));
      checkOpenCLErrors(err, "Set kernel argument 2\n");

      err = clSetKernelArg(bs_kernel_, 3, sizeof(cl_int), 
          static_cast<void *>(&offset));
      checkOpenCLErrors(err, "Set kernel argument 3\n");
      
      // Execute the OpenCL kernel on the list
      err = clEnqueueNDRangeKernel(cmd_queue_, bs_kernel_, CL_TRUE, NULL,
				   globalThreads, localThreads, 0, NULL, &event_);
      checkOpenCLErrors(err, "Enqueue ND Range.\n");
      // clFinish(cmd_queue_);

    } else {
      if (active_cpu_) {
        last_tile_--;
        // fprintf(stderr, "CPU tile: %d \n", last_tile_);
        BlackScholesCPU(rand_array_, call_price_, put_price_,
                        last_tile_ * tile_size_, tile_size_);
      }
    }
  }


  err = clEnqueueReadBuffer(cmd_queue_, d_call_price_, CL_TRUE, 0,
			    done_tiles_ * tile_size_ * sizeof(float),
			    call_price_, 0, NULL,
			    NULL);
  checkOpenCLErrors(err, "Copy data back\n");

  err = clEnqueueReadBuffer(cmd_queue_, d_put_price_, CL_TRUE, 0,
			    done_tiles_ * tile_size_ * sizeof(float),
			    put_price_, 0, NULL,
			    NULL);
  checkOpenCLErrors(err, "Copy data back\n");

  clFinish(cmd_queue_);
}

bool BsCl12Benchmark::IsGpuCompleted() {
  cl_int err;
  cl_int info;

  err = clGetEventInfo(event_, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), (void*)&info, NULL);
  checkOpenCLErrors(err, "Event info\n");
  
  if (info == CL_COMPLETE) return true;
  return false;
}

void BsCl12Benchmark::Cleanup() {

  cl_int ret;

  ret = clReleaseKernel(bs_kernel_);
  ret = clReleaseProgram(program_);
  ret = clReleaseMemObject(d_rand_array_);
  ret = clReleaseMemObject(d_call_price_);
  ret = clReleaseMemObject(d_put_price_);

  checkOpenCLErrors(ret, "Release objects.\n");
    
  BsBenchmark::Cleanup();
}
