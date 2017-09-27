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

#include "src/ga/cl12/ga_cl12_benchmark.h"
//#include "common/cl_util/cl_runtime.h"


#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

void GaCl12Benchmark::Initialize() {
  GaBenchmark::Initialize();

  ClBenchmark::InitializeCl();

  InitializeKernels();
  InitializeBuffers();
  
}


void GaCl12Benchmark::InitializeKernels() {
  cl_int err;
  file_->open("kernels.cl");


  const char *source = file_->getSourceChar();

	//std::cout << source << std::endl;
  
  program_ = clCreateProgramWithSource(context_, 1, (const char **)&source,
				       NULL, &err);
  checkOpenCLErrors(err, "Failed to create program with source...\n");

  //std::cout << "after create with source" << std::endl;
  
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
  

  checkOpenCLErrors(err, "Failed to build program...\n");
  
  ga_kernel_ = clCreateKernel(program_, "ga_cl12", &err);
  checkOpenCLErrors(err, "Failed to create kernel GA\n");
}


void GaCl12Benchmark::InitializeBuffers() {
  GaBenchmark::Initialize();
  coarse_match_result_ = new char[target_sequence_.size()]();
  
  cl_int err;
  d_target_ =
    clCreateBuffer(context_, CL_MEM_READ_WRITE,
		   target_sequence_.size() * sizeof(cl_mem), NULL, &err);
  checkOpenCLErrors(err, "Failed to allocate target buffer");

  d_query_ =
    clCreateBuffer(context_, CL_MEM_READ_WRITE,
		   query_sequence_.size() * sizeof(cl_mem), NULL, &err);
  checkOpenCLErrors(err, "Failed to allocate query buffer");

  d_batch_result_ =
    clCreateBuffer(context_, CL_MEM_READ_WRITE,
		   kBatchSize * sizeof(cl_mem), NULL, &err);
  checkOpenCLErrors(err, "Failed to allocate result buffer");

  err = clEnqueueWriteBuffer(cmd_queue_, d_target_, CL_TRUE, 0,
			     target_sequence_.size() * sizeof(cl_mem), target_sequence_.data(), 0, NULL,
			     NULL);
  checkOpenCLErrors(err, "Copy target data\n");

  err = clEnqueueWriteBuffer(cmd_queue_, d_query_, CL_TRUE, 0,
			     query_sequence_.size() * sizeof(cl_mem), query_sequence_.data(), 0, NULL,
			     NULL);
  checkOpenCLErrors(err, "Copy query data\n");
}

void GaCl12Benchmark::Run() {
  if (collaborative_) {
    CollaborativeRun();
  } else {
    NonCollaborativeRun();
  }
}

void GaCl12Benchmark::CollaborativeRun() {
  cl_int err;
  uint32_t max_searchable_length =
      target_sequence_.size() - coarse_match_length_;
  std::vector<std::thread> threads;
  uint32_t current_position = 0;

  while (current_position < max_searchable_length) {
    char batch_result[kBatchSize] = {0};

    //    cudaMemset(d_batch_result_, 0, kBatchSize);
    cl_uint valToWrite = 0;
    err = clEnqueueFillBuffer(cmd_queue_, d_batch_result_,
			      &valToWrite, sizeof(cl_uint), 0,
			      kBatchSize, 0, NULL, NULL);

    uint32_t end_position = current_position + kBatchSize;
    if (end_position >= max_searchable_length) {
      end_position = max_searchable_length;
    }
    uint32_t length = end_position - current_position;

    size_t localThreads[1] = {64};
    size_t globalThreads[1] = {(length + localThreads[1] - 1) / localThreads[1]};

    //std::cout << "localThreads: " << localThreads[1] << std::endl;
    //std::cout << "globalThreads: " << globalThreads[1] << std::endl;
        
    // Set the arguments of the kernel
    err = clSetKernelArg(ga_kernel_, 0, sizeof(cl_mem),
			 reinterpret_cast<void *>(&d_target_));
    checkOpenCLErrors(err, "Set kernel argument 0\n");

    err = clSetKernelArg(ga_kernel_, 1, sizeof(cl_mem),
			 reinterpret_cast<void *>(&d_query_));
    checkOpenCLErrors(err, "Set kernel argument 1\n");

    err = clSetKernelArg(ga_kernel_, 2, sizeof(cl_mem),
			 reinterpret_cast<void *>(&d_batch_result_));
    checkOpenCLErrors(err, "Set kernel argument 2\n");

    err = clSetKernelArg(ga_kernel_, 3, sizeof(cl_uint),
			 reinterpret_cast<void *>(&length));
    checkOpenCLErrors(err, "Set kernel argument 3\n");

    int qsize = query_sequence_.size();
    err = clSetKernelArg(ga_kernel_, 4, sizeof(cl_uint),
			 reinterpret_cast<void *>(&qsize));
    checkOpenCLErrors(err, "Set kernel argument 4\n");

    err = clSetKernelArg(ga_kernel_, 5, sizeof(cl_uint),
			 reinterpret_cast<void *>(&coarse_match_length_));
    checkOpenCLErrors(err, "Set kernel argument 5\n");

    err = clSetKernelArg(ga_kernel_, 6, sizeof(cl_uint),
			 reinterpret_cast<void *>(&coarse_match_threshold_));
    checkOpenCLErrors(err, "Set kernel argument 6\n");

    err = clSetKernelArg(ga_kernel_, 7, sizeof(cl_uint),
			 reinterpret_cast<void *>(&current_position));
    checkOpenCLErrors(err, "Set kernel argument 7\n");

    // Execute the OpenCL kernel on the list
    err = clEnqueueNDRangeKernel(cmd_queue_, ga_kernel_, CL_TRUE, NULL,
				 globalThreads, localThreads, 0, NULL, NULL);
    checkOpenCLErrors(err, "Enqueue ND Range.\n");
    clFinish(cmd_queue_);

    err = clEnqueueReadBuffer(cmd_queue_, d_batch_result_, CL_TRUE, 0,
			      kBatchSize * sizeof(cl_mem),
			      batch_result, 0, NULL,
			      NULL);
    checkOpenCLErrors(err, "Copy data back\n");

    for (uint32_t i = 0; i < length; i++) {
      if (batch_result[i] != 0) {
        uint32_t end = i + current_position + query_sequence_.size();
        if (end > target_sequence_.size()) end = target_sequence_.size();
        threads.push_back(std::thread(&GaCl12Benchmark::FineMatch, this,
                                      i + current_position, end, &matches_));
      }
    }
    current_position = end_position;
  }
  for (auto &thread : threads) {
    thread.join();
  }
}

void GaCl12Benchmark::NonCollaborativeRun() {
  cl_int err;

  uint32_t max_searchable_length =
      target_sequence_.size() - coarse_match_length_;
  std::vector<std::thread> threads;
  uint32_t current_position = 0;

  while (current_position < max_searchable_length) {
    uint32_t end_position = current_position + kBatchSize;
    if (end_position >= max_searchable_length) {
      end_position = max_searchable_length;
    }
    uint32_t length = end_position - current_position;

    //     cudaMemset(d_batch_result_, 0, kBatchSize);

    cl_uint valToWrite = 0;
    err = clEnqueueFillBuffer(cmd_queue_, d_batch_result_,
			      &valToWrite, sizeof(cl_uint), 0,
			      kBatchSize, 0, NULL, NULL);
        
    // Set the arguments of the kernel
    err = clSetKernelArg(ga_kernel_, 0, sizeof(cl_mem),
			 reinterpret_cast<void *>(&d_target_));
    checkOpenCLErrors(err, "Set kernel argument 0\n");

    err = clSetKernelArg(ga_kernel_, 1, sizeof(cl_mem),
			 reinterpret_cast<void *>(&d_query_));
    checkOpenCLErrors(err, "Set kernel argument 1\n");

    err = clSetKernelArg(ga_kernel_, 2, sizeof(cl_mem),
			 reinterpret_cast<void *>(&d_batch_result_));
    checkOpenCLErrors(err, "Set kernel argument 2\n");

    err = clSetKernelArg(ga_kernel_, 3, sizeof(cl_uint),
			 reinterpret_cast<void *>(&length));
    checkOpenCLErrors(err, "Set kernel argument 3\n");

    int qsize = query_sequence_.size();
    err = clSetKernelArg(ga_kernel_, 4, sizeof(cl_uint),
			 reinterpret_cast<void *>(&qsize));
    checkOpenCLErrors(err, "Set kernel argument 4\n");

    err = clSetKernelArg(ga_kernel_, 5, sizeof(cl_uint),
			 reinterpret_cast<void *>(&coarse_match_length_));
    checkOpenCLErrors(err, "Set kernel argument 5\n");

    err = clSetKernelArg(ga_kernel_, 6, sizeof(cl_uint),
			 reinterpret_cast<void *>(&coarse_match_threshold_));
    checkOpenCLErrors(err, "Set kernel argument 6\n");

    err = clSetKernelArg(ga_kernel_, 7, sizeof(cl_uint),
			 reinterpret_cast<void *>(&current_position));
    checkOpenCLErrors(err, "Set kernel argument 7\n");

    size_t localThreads[1] = {64};
    size_t globalThreads[1] = {(length + localThreads[1] - 1) / localThreads[1]};

    //std::cout << "localThreads: " << localThreads[1] << std::endl;
    //std::cout << "globalThreads: " << globalThreads[1] << std::endl;

    // Execute the OpenCL kernel on the list
    err = clEnqueueNDRangeKernel(cmd_queue_, ga_kernel_, CL_TRUE, NULL,
				 globalThreads, localThreads, 0, NULL, NULL);
    checkOpenCLErrors(err, "Enqueue ND Range.\n");
    
    clFinish(cmd_queue_);

    err = clEnqueueReadBuffer(cmd_queue_, d_batch_result_, CL_TRUE, 0,
			      kBatchSize * sizeof(cl_mem),
			      coarse_match_result_ + current_position, 0, NULL,
			      NULL);
    checkOpenCLErrors(err, "Copy data back\n");

    current_position = end_position;
  }

  for (uint32_t i = 0; i < target_sequence_.size(); i++) {
    if (coarse_match_result_[i] != 0) {
      uint32_t end = i + query_sequence_.size();
      if (end > target_sequence_.size()) end = target_sequence_.size();
      threads.push_back(
          std::thread(&GaCl12Benchmark::FineMatch, this, i, end, &matches_));
    }
  }
  for (auto &thread : threads) {
    thread.join();
  }
}

void GaCl12Benchmark::Cleanup() {
  free(coarse_match_result_);
  GaBenchmark::Cleanup();
}
