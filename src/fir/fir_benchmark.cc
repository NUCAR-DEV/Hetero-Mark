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

#include "src/fir/fir_benchmark.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>

void FirBenchmark::Initialize() {
  num_total_data_ = num_data_per_block_ * num_block_;

  input_ = new float[num_total_data_];
  output_ = new float[num_total_data_];
  coeff_ = new float[num_tap_];

  // unsigned int seed = time(NULL);

  // Initialize input data
  for (unsigned int i = 0; i < num_total_data_; i++) {
    input_[i] = i;
    // input_[i] =
    //     static_cast<float>(rand_r(&seed)) / static_cast<float>(RAND_MAX);
  }

  // Initialize coefficient
  for (unsigned int i = 0; i < num_tap_; i++) {
    coeff_[i] = i;
    // coeff_[i] =
    //     static_cast<float>(rand_r(&seed)) / static_cast<float>(RAND_MAX);
  }
}

void FirBenchmark::Verify() {
  bool has_error = false;
  float *cpu_output = new float[num_total_data_];
  for (unsigned int i = 0; i < num_total_data_; i++) {
    float sum = 0;
    for (unsigned int j = 0; j < num_tap_; j++) {
      if (i < j) continue;
      sum += input_[i - j] * coeff_[j];
    }
    cpu_output[i] = sum;
    if (std::abs(cpu_output[i] - output_[i]) > 1e-5) {
      has_error = true;
      printf("At position %d, expected %f, but was %f.\n", i, cpu_output[i],
             output_[i]);
      exit(-1);
    }
  }

  if (!has_error) {
    printf("Passed! %d data points filtered\n", num_total_data_);
  }

  delete[] cpu_output;
}

void FirBenchmark::Summarize() {
  printf("Input: \n");
  for (unsigned i = 0; i < num_total_data_; i++) {
    printf("%d: %f \n", i, input_[i]);
  }
  printf("\n");

  printf("GPU Output: \n");
  for (unsigned i = 0; i < num_total_data_; i++) {
    printf("%d: %f \n", i, output_[i]);
  }
  printf("\n");
}

void FirBenchmark::Cleanup() {
  delete[] input_;
  delete[] output_;
  delete[] coeff_;
}
