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

#include "src/hist/hip/hist_hip_benchmark.h"

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

__global__ void Histogram(hipLaunchParm lp, uint32_t *pixels,
                          uint32_t *histogram, uint32_t num_colors,
                          uint32_t num_pixels) {
  int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  int gsize = hipGridDim_x * hipBlockDim_x;

  if (tid >= num_pixels) {
    return;
  }

  uint32_t priv_hist[256];
  for (uint32_t i = 0; i < num_colors; i++) {
    priv_hist[i] = 0;
  }

  // Private histogram
  uint32_t index = tid;
  while (index < num_pixels) {
    uint32_t color = pixels[index];
    priv_hist[color]++;
    index += gsize;
  }

  __syncthreads();

  // Copy to global memory
  for (uint32_t i = 0; i < num_colors; i++) {
    if (priv_hist[i] > 0) {
      atomicAdd(histogram + i, priv_hist[i]);
    }
  }
}

void HistHipBenchmark::Initialize() {
  HistBenchmark::Initialize();

  hipMalloc(&d_pixels_, num_pixel_ * sizeof(uint32_t));
  hipMalloc(&d_histogram_, num_color_ * sizeof(uint32_t));
}

void HistHipBenchmark::Run() {
  hipMemcpy(d_pixels_, pixels_, num_pixel_ * sizeof(uint32_t),
            hipMemcpyHostToDevice);
  hipMemset(d_histogram_, 0, num_color_ * sizeof(uint32_t));

  cpu_gpu_logger_->GPUOn();
  hipLaunchKernel(HIP_KERNEL_NAME(Histogram), dim3(8192 / 64), dim3(64), 0, 0,
                  d_pixels_, d_histogram_, num_color_, num_pixel_);
  hipDeviceSynchronize();
  cpu_gpu_logger_->GPUOff();

  hipMemcpy(histogram_, d_histogram_, num_color_ * sizeof(uint32_t),
            hipMemcpyDeviceToHost);
  cpu_gpu_logger_->Summarize();
}

void HistHipBenchmark::Cleanup() {
  hipFree(d_pixels_);
  hipFree(d_histogram_);
  HistBenchmark::Cleanup();
}
