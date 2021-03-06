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

#include "src/hist/hc/hist_hc_benchmark.h"

#include <hcc/hc.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>

void HistHcBenchmark::Initialize() { HistBenchmark::Initialize(); }

void HistHcBenchmark::Run() {
  memset(histogram_, 0, num_color_ * sizeof(uint32_t));

  hc::array_view<uint32_t, 1> av_pixels(num_pixel_, pixels_);
  hc::array_view<uint32_t, 1> av_hist(num_color_, histogram_);
  int num_pixel = num_pixel_;
  int num_color = num_color_;
  int num_wi = 8192;

  cpu_gpu_logger_->GPUOn();
  parallel_for_each(hc::extent<1>(num_wi), [=](hc::index<1> id)[[hc]] {
    uint32_t i = id[0];
    uint32_t local_hist[256] = {0};

    while (i < num_pixel) {
      uint32_t color = av_pixels[i];
      local_hist[color]++;
      i += num_wi;
    }

    for (i = 0; i < num_color; i++) {
      if (local_hist[i] > 0) {
        hc::atomic_fetch_add(av_hist.accelerator_pointer() + i, local_hist[i]);
      }
    }
  });
  cpu_gpu_logger_->GPUOff();
  cpu_gpu_logger_->Summarize();
}

void HistHcBenchmark::Cleanup() { HistBenchmark::Cleanup(); }
