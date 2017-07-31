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

#include "src/ep/hip/ep_hip_benchmark.h"

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <thread>

void EpHipBenchmark::Initialize() {
  EpBenchmark::Initialize();
  hipMalloc(&d_island_, population_ / 2 * sizeof(Creature));
  hipMalloc(&d_fitness_func_, kNumVariables * sizeof(double));
  hipMemcpy(d_fitness_func_, fitness_function_, kNumVariables * sizeof(double),
            hipMemcpyHostToDevice);
}

void EpHipBenchmark::Run() {
  if (pipelined_) {
    PipelinedRun();
  } else {
    NormalRun();
  }
}

void EpHipBenchmark::PipelinedRun() {
  seed_ = kSeedInitValue;
  ReproduceInIsland(&islands_1_);
  for (uint32_t i = 0; i < max_generation_; i++) {
    timer_->Start();
    std::thread t1(&EpHipBenchmark::ReproduceInIsland, this, &islands_2_));
    std::thread t2(&EpHipBenchmark::EvaluateGpu, this, &islands_1_);
    t1.join();
    t2.join();
    timer_->End({"Stage 1"});

    timer_->Start();
    std::thread t3(&EpHipBenchmark::EvaluateGpu, this, &islands_2_);
    std::thread t4(&EpHipBenchmark::SelectInIsland, this, &islands_1_);
    t4.join();
    result_island_1_ = islands_1_[0].fitness;
    std::thread t5(&EpHipBenchmark::CrossoverInIsland, this, &islands_1_);
    t5.join();
    t3.join();
    timer_->End({"Stage 2"});

    timer_->Start();
    std::thread t6(&EpHipBenchmark::SelectInIsland, this, &islands_2_);
    std::thread t7(&EpHipBenchmark::MutateGpu, this, &islands_1_);
    t6.join();
    result_island_2_ = islands_2_[0].fitness;
    std::thread t8(&EpHipBenchmark::CrossoverInIsland, this, &islands_2_);
    t7.join();
    t8.join();
    timer_->End({"Stage 3"});

    timer_->Start();
    std::thread t9(&EpHipBenchmark::MutateGpu, this, &islands_2_);
    std::thread t10(&EpHipBenchmark::ReproduceInIsland, this, &islands_1_);
    t9.join();
    t10.join();
    timer_->End({"Stage 4"});

    timer_->Summarize();
  }
}

void EpHipBenchmark::NormalRun() {
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

__global__ void Evaluate_Kernel(hipLaunchParm lp, Creature *creatures,
                                double *fitness_function, uint32_t count,
                                uint32_t num_vars) {
  uint32_t i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
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

void EpHipBenchmark::EvaluateGpu(std::vector<Creature> *island) {
  hipMemcpy(d_island_, island->data(), population_ / 2 * sizeof(Creature),
            hipMemcpyHostToDevice);
  dim3 block_size(64);
  dim3 grid_size((population_ / 2 * +block_size.x - 1) / block_size.x);
  hipLaunchKernel(HIP_KERNEL_NAME(Evaluate_Kernel), dim3(grid_size),
                  dim3(block_size), 0, 0, d_island_, d_fitness_func_,
                  population_ / 2, kNumVariables);
  hipMemcpy(island->data(), d_island_, population_ / 2 * sizeof(Creature),
            hipMemcpyDeviceToHost);
}

__global__ void Mutate_Kernel(hipLaunchParm lp, Creature *creatures,
                              uint32_t count, uint32_t num_vars) {
  uint32_t i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  if (i >= count) return;

  if (i % 7 != 0) return;
  creatures[i].parameters[i % num_vars] *= 0.5;
}

void EpHipBenchmark::MutateGpu(std::vector<Creature> *island) {
  hipMemcpy(d_island_, island->data(), population_ / 2 * sizeof(Creature),
            hipMemcpyHostToDevice);
  dim3 block_size(64);
  dim3 grid_size((population_ / 2 * +block_size.x - 1) / block_size.x);
  hipLaunchKernel(HIP_KERNEL_NAME(Mutate_Kernel), dim3(grid_size),
                  dim3(block_size), 0, 0, d_island_, population_ / 2,
                  kNumVariables);
  hipMemcpy(island->data(), d_island_, population_ / 2 * sizeof(Creature),
            hipMemcpyDeviceToHost);
}

void EpHipBenchmark::Cleanup() {
  hipFree(d_island_);
  hipFree(d_fitness_func_);
  EpBenchmark::Cleanup();
}
