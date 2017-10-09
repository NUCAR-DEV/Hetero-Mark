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

#include "src/ep/cuda/ep_cuda_benchmark.h"

#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

void EpCudaBenchmark::Initialize() {
  EpBenchmark::Initialize();
  cudaMalloc(&d_island_, population_ / 2 * sizeof(Creature));
  cudaMalloc(&d_fitness_func_, kNumVariables * sizeof(double));
  cudaMemcpy(d_fitness_func_, fitness_function_, kNumVariables * sizeof(double),
             cudaMemcpyHostToDevice);
}

void EpCudaBenchmark::Run() {
  if (pipelined_) {
    PipelinedRun();
  } else {
    NormalRun();
  }
  cpu_gpu_logger_->Summarize();
}

void EpCudaBenchmark::PipelinedRun() {
  seed_ = kSeedInitValue;
  ReproduceInIsland(&islands_1_);
  for (uint32_t i = 0; i < max_generation_; i++) {
    std::thread t1(&EpCudaBenchmark::ReproduceInIsland, this, &islands_2_);
    std::thread t2(&EpCudaBenchmark::EvaluateGpu, this, &islands_1_);
    t1.join();
    t2.join();

    std::thread t3(&EpCudaBenchmark::EvaluateGpu, this, &islands_2_);
    std::thread t4(&EpCudaBenchmark::SelectInIsland, this, &islands_1_);
    t4.join();
    result_island_1_ = islands_1_[0].fitness;
    std::thread t5(&EpCudaBenchmark::CrossoverInIsland, this, &islands_1_);
    t5.join();
    t3.join();

    std::thread t6(&EpCudaBenchmark::SelectInIsland, this, &islands_2_);
    std::thread t7(&EpCudaBenchmark::MutateGpu, this, &islands_1_);
    t6.join();
    result_island_2_ = islands_2_[0].fitness;
    std::thread t8(&EpCudaBenchmark::CrossoverInIsland, this, &islands_2_);
    t7.join();
    t8.join();

    std::thread t9(&EpCudaBenchmark::MutateGpu, this, &islands_2_);
    std::thread t10(&EpCudaBenchmark::ReproduceInIsland, this, &islands_1_);
    t9.join();
    t10.join();
  }
}

void EpCudaBenchmark::NormalRun() {
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

__global__ void Evaluate_Kernel(Creature *creatures, double *fitness_function,
                                uint32_t count, uint32_t num_vars) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;

  double fitness = 0;
  Creature &creature = creatures[i];
  for (int j = 0; j < num_vars; j++) {
    double pow = 1;
    for (int k = 0; k < j + 1; k++) {
      pow *= creature.parameters[j];
    }
    fitness += pow * fitness_function[j];
  }
  creature.fitness = fitness;
}

void EpCudaBenchmark::EvaluateGpu(std::vector<Creature> *island) {
  cudaMemcpy(d_island_, island->data(), population_ / 2 * sizeof(Creature),
             cudaMemcpyHostToDevice);
  dim3 block_size(64);
  dim3 grid_size((population_ / 2 * +block_size.x - 1) / block_size.x);
  cpu_gpu_logger_->GPUOn();
  Evaluate_Kernel<<<grid_size, block_size>>>(d_island_, d_fitness_func_,
                                             population_ / 2, kNumVariables);
  cpu_gpu_logger_->GPUOff();
  cudaMemcpy(island->data(), d_island_, population_ / 2 * sizeof(Creature),
             cudaMemcpyDeviceToHost);
}

__global__ void Mutate_Kernel(Creature *creatures, uint32_t count,
                              uint32_t num_vars) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;

  if (i % 7 != 0) return;
  creatures[i].parameters[i % num_vars] *= 0.5;
}

void EpCudaBenchmark::MutateGpu(std::vector<Creature> *island) {
  cudaMemcpy(d_island_, island->data(), population_ / 2 * sizeof(Creature),
             cudaMemcpyHostToDevice);
  dim3 block_size(64);
  dim3 grid_size((population_ / 2 * +block_size.x - 1) / block_size.x);
  cpu_gpu_logger_->GPUOn();
  Mutate_Kernel<<<grid_size, block_size>>>(d_island_, population_ / 2,
                                           kNumVariables);
  cpu_gpu_logger_->GPUOff();
  cudaMemcpy(island->data(), d_island_, population_ / 2 * sizeof(Creature),
             cudaMemcpyDeviceToHost);
}

void EpCudaBenchmark::Cleanup() {
  cudaFree(d_island_);
  cudaFree(d_fitness_func_);
  EpBenchmark::Cleanup();
}
