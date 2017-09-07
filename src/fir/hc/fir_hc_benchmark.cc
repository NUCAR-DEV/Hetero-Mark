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
 * Author: Shi Dong (shidong@coe.neu.edu)
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

#include "src/fir/hc/fir_hc_benchmark.h"

#include <hc.hpp>

#include <cstdio>
#include <cstdlib>

void FirHcBenchmark::Initialize() {
  FirBenchmark::Initialize();

  // History saves data that carries to next kernel launch
  history_ = new float[num_tap_];
  for (unsigned int i = 0; i < num_tap_; i++) {
    history_[i] = 0.0;
  }
}

void FirHcBenchmark::Run() {
  for (unsigned int i = 0; i < num_tap_; i++) {
    history_[i] = 0.0;
  }

  // hc::array_view<float, 1> av_input(num_total_data_, input_);
  hc::array_view<float, 1> av_coeff(num_tap_, coeff_);
  // hc::array_view<float, 1> av_output(num_total_data_, output_);
  hc::array_view<float, 1> av_history(num_tap_, history_);

  uint32_t num_tap = num_tap_;

  for (unsigned int i = 0; i < num_block_; i++) {
    hc::array_view<float, 1> av_input_sec(num_data_per_block_,
                                          input_ + i * num_data_per_block_);
    hc::array_view<float, 1> av_output_sec(num_data_per_block_,
                                           output_ + i * num_data_per_block_);
    av_output_sec.discard_data();
    hc::extent<1> ex(num_data_per_block_);
    hc::parallel_for_each(ex, [=](hc::index<1> j)[[hc]] {
      float sum = 0;
      for (uint32_t k = 0; k < num_tap; k++) {
        if (j[0] >= k) {
          sum = sum + av_coeff[k] * av_input_sec[j[0] - k];
        } else {
          sum = sum + av_coeff[k] * av_history[num_tap - (k - j[0])];
        }
      }
      av_output_sec[j[0]] = sum;
    });

    av_output_sec.synchronize();

    for (uint32_t i = 0; i < num_tap_; i++) {
      av_history[i] = av_input_sec[num_data_per_block_ - num_tap_ + i];
    }
  }
}

void FirHcBenchmark::Cleanup() {
  FirBenchmark::Cleanup();
  delete[] history_;
}
