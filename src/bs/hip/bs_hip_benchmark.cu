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
 * Author: Yifan Sun (yifansun@coe.neu.edu)
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

#include "src/bs/hip/bs_hip_benchmark.h"

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

__device__ float Phi(float X) {
  float y, absX, t;

  // the coefficients
  const float c1 = 0.319381530f;
  const float c2 = -0.356563782f;
  const float c3 = 1.781477937f;
  const float c4 = -1.821255978f;
  const float c5 = 1.330274429f;

  const float oneBySqrt2pi = 0.398942280f;

  absX = fabs(X);
  t = 1.0f / (1.0f + 0.2316419f * absX);

  y = 1.0f -
      oneBySqrt2pi * exp(-X * X / 2.0f) * t *
          (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))));

  return (X < 0) ? (1.0f - y) : y;
}

__global__ void bs_hip(hipLaunchParm lp, float *rand_array,
                       float *d_call_price_, float *d_put_price_) {
  uint tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

  // the variable representing the value in the array[i]
  float i_rand = rand_array[tid];

  // calculating the initial S,K,T, and R
  float s = 10.0 * i_rand + 100.0 * (1.0f - i_rand);
  float k = 10.0 * i_rand + 100.0 * (1.0f - i_rand);
  float t = 1.0 * i_rand + 10.0 * (1.0f - i_rand);
  float r = 0.01 * i_rand + 0.05 * (1.0f - i_rand);
  float sigma = 0.01 * i_rand + 0.10 * (1.0f - i_rand);

  // Calculating the sigmaSqrtT
  float sigma_sqrt_t_ = sigma * sqrt(t);

  // Calculating the derivatives
  float d1 = (log(s / k) + (r + sigma * sigma / 2.0f) * t) / sigma_sqrt_t_;
  float d2 = d1 - sigma_sqrt_t_;

  // Calculating exponent
  float k_exp_minus_rt_ = k * exp(-r * t);

  // Getting the output call and put prices
  d_call_price_[tid] = s * Phi(d1) - k_exp_minus_rt_ * Phi(d2);
  d_put_price_[tid] = k_exp_minus_rt_ * Phi(-d2) - s * Phi(-d1);
}

void BsHipBenchmark::Initialize() {
  BsBenchmark::Initialize();

  hipMalloc(&d_rand_array_, num_tiles_ * tile_size_ * sizeof(float));
  hipMalloc(&d_call_price_, num_tiles_ * tile_size_ * sizeof(float));
  hipMalloc(&d_put_price_, num_tiles_ * tile_size_ * sizeof(float));

  hipMemcpy(d_rand_array_, rand_array_, num_tiles_ * tile_size_ * sizeof(float),
            hipMemcpyHostToDevice);

  float *temp_call_price = reinterpret_cast<float *>(
      malloc(num_tiles_ * tile_size_ * sizeof(float)));
  float *temp_put_price = reinterpret_cast<float *>(
      malloc(num_tiles_ * tile_size_ * sizeof(float)));
  for (unsigned int i = 0; i < num_tiles_ * tile_size_; i++) {
    temp_call_price[i] = 0;
    temp_put_price[i] = 0;
  }
  hipMemcpy(d_call_price_, temp_call_price,
            num_tiles_ * tile_size_ * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(d_put_price_, temp_put_price,
            num_tiles_ * tile_size_ * sizeof(float), hipMemcpyHostToDevice);
  free(temp_call_price);
  free(temp_put_price);

  hipStreamCreate(&stream_);
}

void BsHipBenchmark::Run() {
  // The main while loop
  uint32_t done_tiles_ = 0;
  uint32_t last_tile_ = num_tiles_;

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

      dim3 block_size(64);
      dim3 grid_size((section_tiles * tile_size_) / 64.00);

      cpu_gpu_logger_->GPUOn();
      hipLaunchKernel(HIP_KERNEL_NAME(bs_hip), dim3(grid_size),
                      dim3(block_size), 0, stream_, d_rand_array_ + offset,
                      d_call_price_ + offset, d_put_price_ + offset);
    } else {
      if (active_cpu_) {
        last_tile_--;
        // fprintf(stderr, "CPU tile: %d \n", last_tile_);
        BlackScholesCPU(rand_array_, call_price_, put_price_,
                        last_tile_ * tile_size_, tile_size_);
      }
    }
  }

  hipDeviceSynchronize();
  cpu_gpu_logger_->GPUOff();

  hipMemcpy(call_price_, d_call_price_,
            done_tiles_ * tile_size_ * sizeof(float), hipMemcpyDeviceToHost);
  hipMemcpy(put_price_, d_put_price_, done_tiles_ * tile_size_ * sizeof(float),
            hipMemcpyDeviceToHost);
  cpu_gpu_logger_->Summarize();
}

bool BsHipBenchmark::IsGpuCompleted() {
  hipError_t ret = hipStreamQuery(stream_);
  if (ret == hipSuccess) {
    cpu_gpu_logger_->GPUOff();
    return true;
  };
  return false;
}

void BsHipBenchmark::Cleanup() {
  hipStreamDestroy(stream_);
  hipFree(d_rand_array_);
  hipFree(d_call_price_);
  hipFree(d_put_price_);
  BsBenchmark::Cleanup();
}
