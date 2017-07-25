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
#include <cstring>
#include "src/pr/cuda/pr_cuda_benchmark.h"

void PrCudaBenchmark::Initialize() {
  PrBenchmark::Initialize();
  page_rank_mtx_1_ = new float[num_nodes_];
  page_rank_mtx_2_ = new float[num_nodes_];
}

__global__ void pr1_cuda(uint32_t *device_row_offsets, uint32_t *device_column_numbers, float *device_values, float *device_mtx_1, float *device_mtx_2)
{
	uint tid = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t initialize = device_row_offsets[tid];
	uint32_t limit = device_row_offsets[tid+1];
	float new_value = 0;
	for(uint32_t j = initialize; j < limit; j++)
	{
		uint32_t index = device_column_numbers[j];
		new_value += device_values[j] * device_mtx_1[index];
	}
	device_mtx_2[tid] = new_value;
}

__global__ void pr2_cuda(uint32_t *device_row_offsets, uint32_t *device_column_numbers, float *device_values, float *device_mtx_1, float *device_mtx_2)
{
	uint tid = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t initialize = device_row_offsets[tid];
        uint32_t limit = device_row_offsets[tid+1];
	float new_value = 0;
	for(uint32_t j = initialize; j < limit; j++)
	{
		uint32_t index = device_column_numbers[j];
		new_value += device_values[j] * device_mtx_2[index];
	}
	device_mtx_1[tid] = new_value;
}

void PrCudaBenchmark::Run() {
  uint32_t i;

  cudaMallocManaged((void**)&device_row_offsets,
                    (num_nodes_ + 1) * sizeof(uint32_t));
  cudaMallocManaged((void**)&device_column_numbers,
                    (num_connections_) * sizeof(uint32_t));
  cudaMallocManaged((void**)&device_values, (num_connections_) * sizeof(float));
  cudaMallocManaged((void**)&device_mtx_1, (num_nodes_) * sizeof(float));
  cudaMallocManaged((void**)&device_mtx_2, (num_nodes_) * sizeof(float));

  memcpy(device_row_offsets, row_offsets_, (num_nodes_ + 1) * sizeof(uint32_t));
  memcpy(device_column_numbers, column_numbers_,
         (num_connections_) * sizeof(uint32_t));
  memcpy(device_values, values_, (num_connections_) * sizeof(float));
  memcpy(device_mtx_1, page_rank_mtx_1_, (num_nodes_) * sizeof(float));
  memcpy(device_mtx_2, page_rank_mtx_2_, (num_nodes_) * sizeof(float));

  dim3 block_size(64);
  dim3 grid_size(num_nodes_ / 64);

  for (i = 0; i < num_nodes_; i++) {
    device_mtx_1[i] = 1.0 / num_nodes_;
  }

  for (i = 0; i < max_iteration_; i++) {
    if (i % 2 == 0) {
      pr1_cuda<<<grid_size, block_size>>>(device_row_offsets,
                                          device_column_numbers, device_values,
                                          device_mtx_1, device_mtx_2);
    }

    else {
      pr2_cuda<<<grid_size, block_size>>>(device_row_offsets,
                                          device_column_numbers, device_values,
                                          device_mtx_1, device_mtx_2);
    }
  }
  cudaDeviceSynchronize();

  for(uint32_t j = 0; j < num_nodes_; j++)
  {
	  page_rank_mtx_1_[j] = device_mtx_1[j];
	  page_rank_mtx_2_[j] = device_mtx_2[j];
  }
  if ( i % 2 != 0)
  {
	  memcpy(page_rank_, page_rank_mtx_1_, num_nodes_ * sizeof(float));
  }
  else
  {
     	  memcpy(page_rank_, page_rank_mtx_2_, num_nodes_ * sizeof(float));
  }
}

 void PrCudaBenchmark::Cleanup() {
    delete[] page_rank_mtx_1_;
    delete[] page_rank_mtx_2_;
    PrBenchmark::Cleanup();
  }
