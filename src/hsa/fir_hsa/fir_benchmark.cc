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

#include <cstdlib>
#include <cstdio>
#include "src/hsa/fir_hsa/kernels.h"
#include "src/hsa/fir_hsa/fir_benchmark.h"

void FirBenchmark::Initialize() {
  num_total_data_ = num_data_ * num_blocks_;

  input_ = new float[num_total_data_];
  output_ = new float[num_total_data_];
  coeff_ = new float[num_tap_];
  history_ = new float[num_tap_];

  unsigned int seed = time(NULL);

  // Initialize input data
  for (unsigned int i = 0; i < num_total_data_; i++) {
    input_[i] = static_cast<float>(rand_r(&seed)) / static_cast<float>(RAND_MAX);
  }

  // Initialize coefficient
  for (unsigned int i = 0; i < num_tap_; i++) {
    coeff_[i] = static_cast<float>(rand_r(&seed)) / static_cast<float>(RAND_MAX);
  }

  // Initialize history
  for (unsigned int i = 0; i < num_tap_; i++) {
    history_[i] = 0.0;
  }

  timer->End({"Initialize"});
  timer->Start();
  FIR_init(0);
  timer->End({"Compile"});
  timer->Start();
}

void FirBenchmark::Run() {
  for (unsigned int i = 0; i < num_blocks_; i++) {
    SNK_INIT_LPARM(lparm, 0);
    lparm->ndim = 1;
    lparm->gdims[0] = num_data_;
    lparm->ldims[0] = 64;
    FIR(input_ + i * num_data_, output_ + i * num_data_, coeff_, history_,
        num_tap_, lparm);
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
    if (abs(cpu_output[i] - output_[i]) > 1e-5) {
      has_error = true;
      printf("At position %d, expected %f, but was %f.\n", i, cpu_output[i],
             output_[i]);
    }
  }

  if (!has_error) {
    printf("Passed! %d data points filtered\n", num_total_data_);
  }

  delete cpu_output;
}

void FirBenchmark::Summarize() {}

void FirBenchmark::Cleanup() {
  delete[] input_;
  delete[] output_;
  delete[] coeff_;
  delete[] history_;
}
