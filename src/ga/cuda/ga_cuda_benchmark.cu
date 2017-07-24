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
#include <cstdio>
#include <cstdlib>
#include <thread>
#include "src/ga/cuda/ga_cuda_benchmark.h"

__global__ void ga_cuda(char *device_target, char *device_query,
                        char *device_batch_result, uint32_t kBatchSize,
                        int query_sequence_length, int coarse_match_length,
                        int coarse_match_threshold, int current_position) {
  uint tid = blockIdx.x * blockDim.x + threadIdx.x;
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

void GaCudaBenchmark::Initialize() {
  GaBenchmark::Initialize();
  coarse_match_result_ = new char[target_sequence_.length()]();

  cudaMallocManaged((void **)&device_target,
                    target_sequence_.length() * sizeof(char));
  cudaMallocManaged((void **)&device_query,
                    query_sequence_.length() * sizeof(char));
  cudaMallocManaged((void **)&device_batch_result, kBatchSize * sizeof(char));

  memcpy(device_target, target_sequence_.c_str(),
         target_sequence_.length() * sizeof(char));
  memcpy(device_query, query_sequence_.c_str(),
         query_sequence_.length() * sizeof(char));
}

void GaCudaBenchmark::Run() {
  int max_searchable_length = target_sequence_.length() - coarse_match_length_;
  std::vector<std::thread> threads;
  int current_position = 0;

  while (current_position < max_searchable_length) {
    char batch_result[kBatchSize] = {0};
    memcpy(device_batch_result, batch_result, kBatchSize * sizeof(char));

    int end_position = current_position + kBatchSize;
    if (end_position >= max_searchable_length) {
      end_position = max_searchable_length;
    }
    int length = end_position - current_position;
    int coarse_match_length = coarse_match_length_;
    int coarse_match_threshold = coarse_match_threshold_;
    int query_sequence_length = query_sequence_.length();

    dim3 block_size(64);
    dim3 grid_size((kBatchSize) / 64.00);

    ga_cuda<<<grid_size, block_size>>>(
        device_target, device_query, device_batch_result, kBatchSize,
        query_sequence_length, coarse_match_length, coarse_match_threshold,
        current_position);
    cudaDeviceSynchronize();
    memcpy(batch_result, device_batch_result, kBatchSize * sizeof(char));
    for (int i = 0; i < length; i++) {
      if (batch_result[i] != 0) {
        unsigned int end = i + current_position + query_sequence_.length();
        if (end > target_sequence_.length()) end = target_sequence_.length();
        threads.push_back(std::thread(&GaCudaBenchmark::FineMatch, this,
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

void GaCudaBenchmark::Cleanup() { GaBenchmark::Cleanup(); }
