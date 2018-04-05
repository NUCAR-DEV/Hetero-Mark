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

#include "src/pr/hip/pr_hip_benchmark.h"

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

void PrHipBenchmark::Initialize() {
  PrBenchmark::Initialize();
  hipMalloc(&device_row_offsets, (num_nodes_ + 1) * sizeof(uint32_t));
  hipMalloc(&device_column_numbers, (num_connections_) * sizeof(uint32_t));
  hipMalloc(&device_values, (num_connections_) * sizeof(float));
  hipMalloc(&device_mtx_1, (num_nodes_) * sizeof(float));
  hipMalloc(&device_mtx_2, (num_nodes_) * sizeof(float));
}

__global__ void pr_hip(hipLaunchParm lp, uint32_t *device_row_offsets,
                       uint32_t *device_column_numbers, float *device_values,
                       float *device_mtx_1, float *device_mtx_2) {
  uint tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  uint32_t initialize = device_row_offsets[tid];
  uint32_t limit = device_row_offsets[tid + 1];
  float new_value = 0;
  for (uint32_t j = initialize; j < limit; j++) {
    uint32_t index = device_column_numbers[j];
    new_value += device_values[j] * device_mtx_1[index];
  }
  device_mtx_2[tid] = new_value;
}

void PrHipBenchmark::Run() {
  uint32_t i;

  hipMemcpy(device_row_offsets, row_offsets_,
            (num_nodes_ + 1) * sizeof(uint32_t), hipMemcpyHostToDevice);
  hipMemcpy(device_column_numbers, column_numbers_,
            (num_connections_) * sizeof(uint32_t), hipMemcpyHostToDevice);
  hipMemcpy(device_values, values_, (num_connections_) * sizeof(float),
            hipMemcpyHostToDevice);

  dim3 block_size(64);
  dim3 grid_size(num_nodes_ / 64);

  float *temp_mtx =
      reinterpret_cast<float *>(malloc(num_nodes_ * sizeof(float)));
  for (i = 0; i < num_nodes_; i++) {
    temp_mtx[i] = 1.0 / num_nodes_;
  }
  hipMemcpy(device_mtx_1, temp_mtx, num_nodes_ * sizeof(float),
            hipMemcpyHostToDevice);
  free(temp_mtx);

  cpu_gpu_logger_->GPUOn();
  for (i = 0; i < max_iteration_; i++) {
    if (i % 2 == 0) {
      hipLaunchKernel(HIP_KERNEL_NAME(pr_hip), dim3(grid_size),
                      dim3(block_size), 0, 0, device_row_offsets,
                      device_column_numbers, device_values, device_mtx_1,
                      device_mtx_2);
    } else {
      hipLaunchKernel(HIP_KERNEL_NAME(pr_hip), dim3(grid_size),
                      dim3(block_size), 0, 0, device_row_offsets,
                      device_column_numbers, device_values, device_mtx_2,
                      device_mtx_1);
    }
  }

  if (i % 2 != 0) {
    hipMemcpy(page_rank_, device_mtx_1, num_nodes_ * sizeof(float),
              hipMemcpyDeviceToHost);
  } else {
    hipMemcpy(page_rank_, device_mtx_2, num_nodes_ * sizeof(float),
              hipMemcpyDeviceToHost);
  }
  cpu_gpu_logger_->GPUOff();
  cpu_gpu_logger_->Summarize();
}

void PrHipBenchmark::Cleanup() {
  hipFree(device_row_offsets);
  hipFree(device_column_numbers);
  hipFree(device_values);
  hipFree(device_mtx_1);
  hipFree(device_mtx_2);

  PrBenchmark::Cleanup();
}
