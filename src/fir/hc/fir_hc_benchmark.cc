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

  mem_manager = new ArrayMemoryManager();

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

  for (unsigned int i = 0; i < num_total_data_; i++) {
    output_[i] = 0.0;
  }

  auto dmem_coeff = mem_manager->Shadow(coeff_, num_tap_ * sizeof(float));
  auto dmem_history = mem_manager->Shadow(history_, num_tap_ * sizeof(float));

  float *d_coeff = static_cast<float *>(dmem_coeff->GetDevicePtr());
  float *d_history = static_cast<float *>(dmem_history->GetDevicePtr());

  uint32_t num_tap = num_tap_;

  for (unsigned int i = 0; i < num_block_; i++) {
    float *h_input = input_ + i * num_data_per_block_;
    float *h_output = output_ + i * num_data_per_block_;

    auto dmem_input =
        mem_manager->Shadow(h_input, num_data_per_block_ * sizeof(float));
    auto dmem_output =
        mem_manager->Shadow(h_output, num_data_per_block_ * sizeof(float));
    dmem_input->HostToDevice();
    float *d_input = static_cast<float *>(dmem_input->GetDevicePtr());
    float *d_output = static_cast<float *>(dmem_output->GetDevicePtr());

    printf("Here %d, %d\n", i, num_block_);
    // printf("%p, %p\n", dmem_input->GetDevicePtr(), dmem_input->GetHostPtr());

    hc::extent<1> ex(num_data_per_block_);
    auto future = hc::parallel_for_each(ex, [=](hc::index<1> j)[[hc]] {
      float sum = 0;

      for (uint32_t k = 0; k < num_tap; k++) {
        if (j[0] >= k) {
          sum = sum + d_coeff[k] * d_input[j[0] - k];
        } else {
          sum = sum + d_coeff[k] * d_history[num_tap - (k - j[0])];
        }
      }
      d_output[j[0]] = sum;
    });
    future.wait();

    dmem_output->DeviceToHost();

    for (uint32_t j = 0; j < num_tap_; j++) {
      history_[j] = h_input[num_data_per_block_ - num_tap_ + j];
    }
    dmem_history->HostToDevice();

    printf("Here %d, %d\n", i, num_block_);
  }
}

void FirHcBenchmark::Cleanup() {
  FirBenchmark::Cleanup();
  delete[] history_;
}
