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

#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include "src/bs/bs_benchmark.h"

float BsBenchmark::Phi(float X) {
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

void BsBenchmark::Initialize() {
  // We fit the square root in a WG of 64
  num_tiles_ = (((num_elements_ - 1) / GetWorkGroupSize()) + 1);
  tile_size_ = GetWorkGroupSize();

  // We set our Random Array as a square matrix based on the input.
  // We are adding the 4 for float 4
  rand_array_ = new float[num_tiles_ * tile_size_];

  // We have to check for viability of our random array since the size
  // might exceed the allowed size for an array
  if (rand_array_ == nullptr) {
    std::cout << " Failed to create the input array. This might "
                 "be related to large size of the array\n";
    exit(1);
  }

  // seed -- for ease of use and recreation purposes
  // we take the num_elements as seed
  uint32_t seed = num_elements_;
  // Populate the random array with random variables
  for (uint32_t i = 0; i < num_tiles_ * tile_size_; i++) {
    rand_array_[i] =
        static_cast<float>(rand_r(&seed)) / static_cast<float>(RAND_MAX);
  }

  // setup the call Price array
  call_price_ = new float[num_tiles_ * tile_size_];

  // setup the put price array
  put_price_ = new float[num_tiles_ * tile_size_];
}

void BsBenchmark::BlackScholesCPU(float *rand_array, float *call_price,
                                  float *put_price, uint32_t index,
                                  uint32_t extent) {
  for (uint32_t local_index = 0; local_index < extent; ++local_index) {
    // set the y to be the golobal index
    uint32_t y = index + local_index;

    // Set the current float of the target random array
    float i_rand = rand_array[y];

    // calculation of the float 4, one float at a time
    float s = kSLowerLimit * i_rand + kSUpperLimit * (1.0f - i_rand);
    float k = kKLowerLimit * i_rand + kKUpperLimit * (1.0f - i_rand);
    float t = kTLowerLimit * i_rand + kTUpperLimit * (1.0f - i_rand);
    float r = kRLowerLimit * i_rand + kRUpperLimit * (1.0f - i_rand);
    float sigma =
        kSigmaLowerLimit * i_rand + kSigmaUpperLimit * (1.0f - i_rand);

    // Calculating the sigmaSqrtT
    float sigma_sqrt_t_ = sigma * sqrt(t);

    // Calculating the derivatives
    float d1 = (log(s / k) + (r + sigma * sigma / 2.0f) * t) / sigma_sqrt_t_;
    float d2 = d1 - sigma_sqrt_t_;

    // Calculating exponent
    float k_exp_minus_rt_ = k * exp(-r * t);

    // setting the call and price
    call_price[y] = s * Phi(d1) - k_exp_minus_rt_ * Phi(d2);
    put_price[y] = k_exp_minus_rt_ * Phi(-d2) - s * Phi(-d1);
  }
}

void BsBenchmark::Verify() {
  // For every elements in the array we will run the
  // following sequence of equations
  float *verify_call_price_ = new float[num_tiles_ * tile_size_];
  float *verify_put_price_ = new float[num_tiles_ * tile_size_];

  // The main loop only runs the CPU code on the sections
  for (uint32_t section = 0; section < num_tiles_; section++) {
    // First we will get the index to the section
    uint32_t index = section * tile_size_;

    // We can the CPU code to run the BS on the section
    BlackScholesCPU(rand_array_, verify_call_price_, verify_put_price_, index,
                    tile_size_);
  }
  fprintf(stderr, "[512]= %f\n", call_price_[512]);

  // Here we verify the results
  for (uint32_t y = 0; y < num_tiles_ * tile_size_; ++y) {
    if (fabs(verify_call_price_[y] - call_price_[y]) > 1e-4f) {
      std::cerr << "Verification failed. Call Price Position " << y
                << ": Expected to be " << std::fixed << std::setprecision(4)
                << verify_call_price_[y] << "but it is" << std::fixed
                << std::setprecision(4) << call_price_[y] << std::endl;
      return;
    }
    if (fabs(verify_put_price_[y] - put_price_[y]) > 1e-4f) {
      std::cerr << "Verification failed. Put Price Position " << y
                << ": Expected to be " << std::fixed << std::setprecision(4)
                << verify_put_price_[y] << "but it is" << std::fixed
                << std::setprecision(4) << put_price_[y] << std::endl;
      return;
    }
  }
  std::cout << "Passed." << std::endl;

  // Delete the verification structures
  delete[] verify_call_price_;
  delete[] verify_put_price_;
}

void BsBenchmark::Summarize() {
  // Print the input size
  std::cout << "The number of elements are extended from " << num_elements_
            << " to " << num_tiles_ *tile_size_ << " so it fits in blocks of "
            << GetWorkGroupSize() << std::endl;

  // Print the input random elements
  std::cout << "Random Array:" << std::endl;
  for (uint32_t i = 0; i < num_tiles_ * tile_size_; ++i)
    std::cout << rand_array_[i] << " ";
  std::cout << std::endl;

  // Print the output results for Call Price
  std::cout << "Call Price:" << std::endl;
  for (uint32_t i = 0; i < num_tiles_ * tile_size_; ++i)
    std::cout << call_price_[i] << " ";
  std::cout << std::endl;

  // Print the output results for Put Price
  std::cout << "Put Price:" << std::endl;
  for (uint32_t i = 0; i < num_tiles_ * tile_size_; ++i)
    std::cout << put_price_[i] << " ";
  std::cout << std::endl;
}

void BsBenchmark::Cleanup() {
  // Delete the array of elements
  delete rand_array_;

  // Delete the result arrays
  delete call_price_;
  delete put_price_;
}
