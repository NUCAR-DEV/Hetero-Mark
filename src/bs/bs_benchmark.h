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

#ifndef SRC_BS_BS_BENCHMARK_H_
#define SRC_BS_BS_BENCHMARK_H_

#include "src/common/benchmark/benchmark.h"
#include "src/common/time_measurement/time_measurement.h"

class BsBenchmark : public Benchmark {
 protected:
  // fixed constants, used in the calculations
  const float kSLowerLimit = 10.0;
  const float kSUpperLimit = 100.0;
  const float kKLowerLimit = 10.0;
  const float kKUpperLimit = 100.0;
  const float kTLowerLimit = 1.0;
  const float kTUpperLimit = 10.0;
  const float kRLowerLimit = 0.01;
  const float kRUpperLimit = 0.05;
  const float kSigmaLowerLimit = 0.01;
  const float kSigmaUpperLimit = 0.10;

  // Number of elements passed as input by user
  uint32_t num_elements_;

  // The argument for activating CPU for compute
  bool active_cpu_;

  // GPU chunk size
  uint32_t gpu_chunk_;

  // number of tiles of the final array, and size of tiles
  uint32_t num_tiles_;
  uint32_t tile_size_;

  // Shared buffer for the random array
  float* rand_array_;

  // The put price and call price
  float* put_price_;
  float* call_price_;

  //
  // member function
  //
  // Phi is the cumulative normal distribution function
  float Phi(float X);

  /**
   * The CPU code for running the BS on a tile
   */
  void BlackScholesCPU(float* rand_array, float* call_price, float* put_price,
                       uint32_t index, uint32_t extent);

 public:
  BsBenchmark() : Benchmark() {}
  void Initialize() override;
  void Run() override{};
  void Verify() override;
  void Summarize() override;
  void Cleanup() override;

  // Setters
  void SetNumElements(uint32_t num_elements) { num_elements_ = num_elements; }

  // Setter for has CPU
  void SetActiveCPU(bool active_cpu) { active_cpu_ = active_cpu; }

  // Setter for the GPU chunk size
  void SetGpuChunk(uint32_t gpu_chunk) {
    gpu_chunk_ = (gpu_chunk != 0) ? gpu_chunk : GetNumComputeUnits();
  }
};

#endif  // SRC_BS_BS_BENCHMARK_H_
