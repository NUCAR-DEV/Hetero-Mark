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

#include "src/ep/cl12/ep_cl12_benchmark.h"

#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

void EpCl12Benchmark::Initialize() {
  EpBenchmark::Initialize();

  ClBenchmark::InitializeCl();
  InitializeKernels();

  InitializeBuffers();
}

void EpCl12Benchmark::InitializeKernels() {
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

  Evaluate_Kernel_ = clCreateKernel(program_, "Evaluate_Kernel", &err);
  checkOpenCLErrors(err, "Failed to create evaluate kernel\n");

  Mutate_Kernel_ = clCreateKernel(program_, "Mutate_Kernel", &err);
  checkOpenCLErrors(err, "Failed to create mutate kernel\n");
}

void EpCl12Benchmark::InitializeBuffers() {

  // Create memory buffers for background, foreground and frames on the device
  cl_int err;

  d_island_=
    clCreateBuffer(context_, CL_MEM_READ_WRITE,
		   population_ / 2 * sizeof(Creature), NULL, &err);
  checkOpenCLErrors(err, "Failed to allocate island buffer");
  
  d_fitness_func_ = clCreateBuffer(context_, CL_MEM_READ_WRITE,
				   kNumVariables * sizeof(cl_mem), NULL, &err);
  checkOpenCLErrors(err, "Failed to allocate fitness function buffer");

  err = clEnqueueWriteBuffer(cmd_queue_, d_fitness_func_, CL_TRUE, 0,
			     kNumVariables * sizeof(cl_mem), fitness_function_, 0, NULL, NULL);

  checkOpenCLErrors(err, "Copy fitness function\n");
}


void EpCl12Benchmark::Run() {
  if (pipelined_) {
    PipelinedRun();
  } else {
    NormalRun();
  }
}

void EpCl12Benchmark::PipelinedRun() {
  seed_ = kSeedInitValue;
  ReproduceInIsland(&islands_1_);
  for (uint32_t i = 0; i < max_generation_; i++) {
    timer_->Start();
    std::thread t1(&EpCl12Benchmark::ReproduceInIsland, this, &islands_2_);
    std::thread t2(&EpCl12Benchmark::EvaluateGpu, this, &islands_1_);
    t1.join();
    t2.join();
    timer_->End({"Stage 1"});

    timer_->Start();
    std::thread t3(&EpCl12Benchmark::EvaluateGpu, this, &islands_2_);
    std::thread t4(&EpCl12Benchmark::SelectInIsland, this, &islands_1_);
    t4.join();
    result_island_1_ = islands_1_[0].fitness;
    std::thread t5(&EpCl12Benchmark::CrossoverInIsland, this, &islands_1_);
    t5.join();
    t3.join();
    timer_->End({"Stage 2"});

    timer_->Start();
    std::thread t6(&EpCl12Benchmark::SelectInIsland, this, &islands_2_);
    std::thread t7(&EpCl12Benchmark::MutateGpu, this, &islands_1_);
    t6.join();
    result_island_2_ = islands_2_[0].fitness;
    std::thread t8(&EpCl12Benchmark::CrossoverInIsland, this, &islands_2_);
    t7.join();
    t8.join();
    timer_->End({"Stage 3"});

    timer_->Start();
    std::thread t9(&EpCl12Benchmark::MutateGpu, this, &islands_2_);
    std::thread t10(&EpCl12Benchmark::ReproduceInIsland, this, &islands_1_);
    t9.join();
    t10.join();
    timer_->End({"Stage 4"});

    timer_->Summarize();
  }
}

void EpCl12Benchmark::NormalRun() {
  seed_ = kSeedInitValue;
  for (uint32_t i = 0; i < max_generation_; i++) {
    Reproduce();
    EvaluateGpu(&islands_1_);
    EvaluateGpu(&islands_2_);
    Select();

    result_island_1_ = islands_1_[0].fitness;
    result_island_2_ = islands_2_[0].fitness;

    Crossover();
    MutateGpu(&islands_1_);
    MutateGpu(&islands_2_);
  }
}

void EpCl12Benchmark::EvaluateGpu(std::vector<Creature> *island) {
  cl_int ret;

  ret = clEnqueueWriteBuffer(cmd_queue_, d_island_, CL_TRUE, 0,
			     population_ / 2 * sizeof(Creature), island->data(), 0, NULL, NULL);
  checkOpenCLErrors(ret, "Copy island data\n");

  size_t localThreads[1] = {64};
  size_t globalThreads[1] = {(population_ / 2 * + localThreads[1] - 1) / localThreads[1]};

  // Set kernel arguments
  ret = clSetKernelArg(Evaluate_Kernel_, 0, sizeof(Creature),
		       reinterpret_cast<void *>(&d_island_));
  checkOpenCLErrors(ret, "Set kernel argument 0\n");

  ret = clSetKernelArg(Evaluate_Kernel_, 1, sizeof(cl_mem),
		       reinterpret_cast<void *>(&d_fitness_func_));
  checkOpenCLErrors(ret, "Set kernel argument 1\n");

  uint32_t half_population_ = population_ / 2;
  
  ret = clSetKernelArg(Evaluate_Kernel_, 2, sizeof(cl_uint),
		       reinterpret_cast<void *>(&half_population_));
  checkOpenCLErrors(ret, "Set kernel argument 2\n");

  
  ret = clSetKernelArg(Evaluate_Kernel_, 3, sizeof(cl_uint),
		       const_cast<void*>(reinterpret_cast<const void *>(&kNumVariables)));
  checkOpenCLErrors(ret, "Set kernel argument 3\n");

  // Launch kernel
  ret = clEnqueueNDRangeKernel(cmd_queue_, Evaluate_Kernel_, CL_TRUE, NULL,
			       globalThreads, localThreads, 0, NULL, NULL);
  checkOpenCLErrors(ret, "Enqueue ND Range.\n");
  clFinish(cmd_queue_);

  // Get data back
  ret = clEnqueueReadBuffer(cmd_queue_, d_island_, CL_TRUE, 0,
			    population_ / 2 * sizeof(Creature),
			    island->data(), 0, NULL,
			    NULL);
  checkOpenCLErrors(ret, "Copy data back\n");
  clFinish(cmd_queue_);
}

void EpCl12Benchmark::MutateGpu(std::vector<Creature> *island) {
  cl_int ret;
  
  ret = clEnqueueWriteBuffer(cmd_queue_, d_island_, CL_TRUE, 0,
			     population_ / 2 * sizeof(cl_mem), island->data(), 0, NULL, NULL);
  checkOpenCLErrors(ret, "Copy island data\n");

  size_t localThreads[1] = {64};
  size_t globalThreads[1] = {(population_ / 2 * + localThreads[1] - 1) / localThreads[1]};

// Set kernel arguments
  ret = clSetKernelArg(Mutate_Kernel_, 0, sizeof(cl_mem),
		       reinterpret_cast<void *>(&d_island_));
  checkOpenCLErrors(ret, "Set kernel argument 0\n");

  uint32_t half_population_ = population_ / 2;
  
  ret = clSetKernelArg(Mutate_Kernel_, 1, sizeof(cl_uint),
		       reinterpret_cast<void *>(&half_population_));
  checkOpenCLErrors(ret, "Set kernel argument 1\n");

  ret = clSetKernelArg(Mutate_Kernel_, 2, sizeof(cl_uint),
		       const_cast<void*>(reinterpret_cast<const void *>(&kNumVariables)));
  checkOpenCLErrors(ret, "Set kernel argument 2\n");
  
  // Launch kernel
  ret = clEnqueueNDRangeKernel(cmd_queue_, Mutate_Kernel_, CL_TRUE, NULL,
			       globalThreads, localThreads, 0, NULL, NULL);
  checkOpenCLErrors(ret, "Enqueue ND Range.\n");
  clFinish(cmd_queue_);

  // Get data back
  ret = clEnqueueReadBuffer(cmd_queue_, d_island_, CL_TRUE, 0,
			    population_ / 2 * sizeof(cl_mem),
			    island->data(), 0, NULL,
			    NULL);
  checkOpenCLErrors(ret, "Copy data back\n");
  clFinish(cmd_queue_);
}

void EpCl12Benchmark::Cleanup() {
  cl_int ret;

  ret = clReleaseKernel(Evaluate_Kernel_);
  ret = clReleaseKernel(Mutate_Kernel_);
  ret = clReleaseProgram(program_);
  ret = clReleaseMemObject(d_island_);
  ret = clReleaseMemObject(d_fitness_func_);

  checkOpenCLErrors(ret, "Release objects.\n");

  EpBenchmark::Cleanup();
}
