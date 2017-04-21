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

#include "src/bs/hc/bs_hc_benchmark.h"
#include <hcc/hc.hpp>
#include <hcc/hc_math.hpp>
#include <cstdlib>
#include <cstdio>
#include <ctime>

void BsHcBenchmark::Initialize() { BsBenchmark::Initialize(); }

float BsHcBenchmark::Phi(float X) [[hc]] {
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

void BsHcBenchmark::Run() {
  // Provide the array view of 1-Dimension float array (similar to
  // the CPU code of the verification stage)
  hc::array_view<float, 1> rand_array(num_tiles_ * tile_size_, rand_array_);

  hc::array_view<float, 1> av_call_price(num_tiles_ * tile_size_, call_price_);
  hc::array_view<float, 1> av_put_price(num_tiles_ * tile_size_, put_price_);

  // We create the GPU arrays too, but if cpu is not active, they are not
  // used
  hc::array<float, 1> call_(num_tiles_ * tile_size_);
  hc::array<float, 1> put_(num_tiles_ * tile_size_);
  if (active_cpu_) {
    hc::array_view<float, 1> shallow_call_(call_);
    hc::array_view<float, 1> shallow_put_(put_);
    // The output call_price is replaced with the call array if
    // CPU is active
    av_call_price = shallow_call_;
    av_put_price = shallow_put_;
  }
  // the HC future completion object
  hc::completion_future fut;

  // the boolean object for the first lunch
  bool first_launch_ = true;

  // The main while loop
  uint32_t done_tiles_ = 0;
  uint32_t last_tile_ = num_tiles_;

  // while the done tiles are less than num_tiles, continue
  while (done_tiles_ < last_tile_) {
    // First check to make sure that we are launching the first set
    if (first_launch_ || fut.is_ready()) {
      // No longer the first lunch after this point so
      // turn it off
      first_launch_ = false;

      // Set the size of the section based on the number of tiles
      // and the number of compute units
      uint32_t section_tiles = (gpu_chunk_ < last_tile_ - done_tiles_)
                                   ? gpu_chunk_
                                   : last_tile_ - done_tiles_;

      // the section of the random array we care about
      hc::array_view<float, 1> section = rand_array.section(
          done_tiles_ * tile_size_, section_tiles * tile_size_);

      // The section of the call output we will write in
      hc::array_view<float, 1> av_call_price_section = av_call_price.section(
          done_tiles_ * tile_size_, section_tiles * tile_size_);

      // The section of the put output we will write in
      hc::array_view<float, 1> av_put_price_section = av_put_price.section(
          done_tiles_ * tile_size_, section_tiles * tile_size_);

      // GPU is running the following tiles
      fprintf(stderr, "GPU tiles: %d to %d\n", done_tiles_,
              done_tiles_ + section_tiles);
      done_tiles_ += section_tiles;

      // Convert member var to local var to use in the kenrel
      float sLowerLimit = kSLowerLimit;
      float kLowerLimit = kKLowerLimit;
      float tLowerLimit = kTLowerLimit;
      float rLowerLimit = kRLowerLimit;
      float sigmaLowerLimit = kSigmaLowerLimit;
      float sUpperLimit = kSUpperLimit;
      float kUpperLimit = kKUpperLimit;
      float tUpperLimit = kTUpperLimit;
      float rUpperLimit = kRUpperLimit;
      float sigmaUpperLimit = kSigmaUpperLimit;

      // Run the application
      fut = hc::parallel_for_each(hc::extent<1>(section_tiles * tile_size_),
                                  [=](hc::index<1> index)[[hc]] {
        // the variable representing the value in the array[i]
        float i_rand = section[index];

        // calculating the initial S,K,T, and R
        float s = sLowerLimit * i_rand + sUpperLimit * (1.0f - i_rand);
        float k = kLowerLimit * i_rand + kUpperLimit * (1.0f - i_rand);
        float t = tLowerLimit * i_rand + tUpperLimit * (1.0f - i_rand);
        float r = rLowerLimit * i_rand + rUpperLimit * (1.0f - i_rand);
        float sigma =
            sigmaLowerLimit * i_rand + sigmaUpperLimit * (1.0f - i_rand);
        //
        // Calculating the sigmaSqrtT
        float sigma_sqrt_t = sigma * sqrt(t);

        // Calculating the derivatives
        float d1 =
            (log(s / k) + (r + sigma * sigma / 2.0f) * t) / sigma_sqrt_t;
        float d2 = d1 - sigma_sqrt_t;
        //
        // Calculating exponent
        float k_exp_minus_rt = k * exp(-r * t);

        // Getting the output call and put prices
        av_call_price_section[index] = s * Phi(d1) - k_exp_minus_rt * Phi(d2);
        av_put_price_section[index] = k_exp_minus_rt * Phi(-d2) - s * Phi(-d1);
      });
    } else {
      // CPU portion of the code will run consecutive code until the
      // GPU becomes ready
      if (active_cpu_) {
        last_tile_--;
        fprintf(stderr, "CPU tile: %d \n", last_tile_);
        BlackScholesCPU(rand_array_, call_price_, put_price_,
                        last_tile_ * tile_size_, tile_size_);
      }
    }
  }
  fut.wait();
  if (active_cpu_) {
    hc::array_view<float, 1> call_partial_ =
        av_call_price.section(0, done_tiles_ * tile_size_);
    hc::array_view<float, 1> put_partial_ =
        av_put_price.section(0, done_tiles_ * tile_size_);

    hc::copy(call_partial_, call_price_);
    hc::copy(put_partial_, put_price_);
  }
}

void BsHcBenchmark::Cleanup() { BsBenchmark::Cleanup(); }
