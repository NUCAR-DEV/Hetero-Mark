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

#include "src/ga/hip/ga_hip_benchmark.h"

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

__global__ void ga_hip(hipLaunchParm lp, char *device_target,
                       char *device_query, char *device_batch_result,
                       uint32_t length, int query_sequence_length,
                       int coarse_match_length, int coarse_match_threshold,
                       int current_position) {
  uint tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  if (tid > length) return;
  bool match = false;
  int max_length = query_sequence_length - coarse_match_length;

  for (uint32_t i = 0; i <= max_length; i++) {
    int distance = 0;
    for (int j = 0; j < coarse_match_length; j++) {
      if (device_target[current_position + tid + j] != device_query[i + j]) {
        distance++;
      }
    }

    if (distance < coarse_match_threshold) {
      match = true;
      break;
    }
  }
  if (match) {
    device_batch_result[tid] = 1;
  }
}

void GaHipBenchmark::Initialize() {
  GaBenchmark::Initialize();
  coarse_match_result_ = new char[target_sequence_.size()]();

  hipMalloc(&d_target_, target_sequence_.size() * sizeof(char));
  hipMalloc(&d_query_, query_sequence_.size() * sizeof(char));
  hipMalloc(&d_batch_result_, kBatchSize * sizeof(char));

  hipMemcpy(d_target_, target_sequence_.data(),
            target_sequence_.size() * sizeof(char), hipMemcpyHostToDevice);
  hipMemcpy(d_query_, query_sequence_.data(),
            query_sequence_.size() * sizeof(char), hipMemcpyHostToDevice);
}

void GaHipBenchmark::Run() {
  if (collaborative_) {
    CollaborativeRun();
  } else {
    NonCollaborativeRun();
  }
}

void GaHipBenchmark::CollaborativeRun() {
  uint32_t max_searchable_length =
      target_sequence_.size() - coarse_match_length_;
  std::vector<std::thread> threads;
  uint32_t current_position = 0;

  while (current_position < max_searchable_length) {
    char batch_result[kBatchSize] = {0};
    hipMemset(d_batch_result_, 0, kBatchSize);

    uint32_t end_position = current_position + kBatchSize;
    if (end_position >= max_searchable_length) {
      end_position = max_searchable_length;
    }
    uint32_t length = end_position - current_position;

    dim3 block_size(64);
    dim3 grid_size((length + block_size.x - 1) / block_size.x);

    hipLaunchKernel(HIP_KERNEL_NAME(ga_hip), dim3(grid_size), dim3(block_size),
                    0, 0, d_target_, d_query_, d_batch_result_, length,
                    query_sequence_.size(), coarse_match_length_,
                    coarse_match_threshold_, current_position);
    hipDeviceSynchronize();
    hipMemcpy(batch_result, d_batch_result_, kBatchSize * sizeof(char),
              hipMemcpyDeviceToHost);

    for (uint32_t i = 0; i < length; i++) {
      if (batch_result[i] != 0) {
        uint32_t end = i + current_position + query_sequence_.size();
        if (end > target_sequence_.size()) end = target_sequence_.size();
        threads.push_back(std::thread(&GaHipBenchmark::FineMatch, this,
                                      i + current_position, end,
                                      std::ref(matches_)));
      }
    }
    current_position = end_position;
  }
  for (auto &thread : threads) {
    thread.join();
  }
}

void GaHipBenchmark::NonCollaborativeRun() {
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

    dim3 block_size(64);
    dim3 grid_size((length + block_size.x - 1) / block_size.x);

    hipMemset(d_batch_result_, 0, kBatchSize);

    hipLaunchKernel(HIP_KERNEL_NAME(ga_hip), dim3(grid_size), dim3(block_size),
                    0, 0, d_target_, d_query_, d_batch_result_, length,
                    query_sequence_.size(), coarse_match_length_,
                    coarse_match_threshold_, current_position);
    hipDeviceSynchronize();
    hipMemcpy(coarse_match_result_ + current_position, d_batch_result_,
              kBatchSize * sizeof(char), hipMemcpyDeviceToHost);
    current_position = end_position;
  }

  for (uint32_t i = 0; i < target_sequence_.size(); i++) {
    if (coarse_match_result_[i] != 0) {
      uint32_t end = i + query_sequence_.size();
      if (end > target_sequence_.size()) end = target_sequence_.size();
      threads.push_back(std::thread(&GaHipBenchmark::FineMatch, this, i, end,
                                    std::ref(matches_)));
    }
  }
  for (auto &thread : threads) {
    thread.join();
  }
}

void GaHipBenchmark::Cleanup() {
  free(coarse_match_result_);
  GaBenchmark::Cleanup();
}
