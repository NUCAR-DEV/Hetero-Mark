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

#include "src/pr/hc/pr_hc_benchmark.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <hcc/hc.hpp>

void PrHcBenchmark::Initialize() {
  PrBenchmark::Initialize();
  page_rank_mtx_1_ = new float[num_nodes_];
  page_rank_mtx_2_ = new float[num_nodes_];
}

void PrHcBenchmark::Run() {
  uint32_t i;

  hc::array_view<uint32_t, 1> av_row_offsets(num_nodes_ + 1, row_offsets_);
  hc::array_view<uint32_t, 1> av_column_numbers(num_connections_,
                                                column_numbers_);
  hc::array_view<float, 1> av_values(num_connections_, values_);
  // TODO: We can use the output vector as mtx_1
  hc::array_view<float, 1> av_mtx_1(num_nodes_, page_rank_mtx_1_);
  hc::array_view<float, 1> av_mtx_2(num_nodes_, page_rank_mtx_2_);

  // TODO: this can be converted to a kernel
  for (i = 0; i < num_nodes_; i++) {
    av_mtx_1[i] = 1.0 / num_nodes_;
  }

  for (i = 0; i < max_iteration_; i++) {
    if (i % 2 == 0) {
      parallel_for_each(hc::extent<1>(num_nodes_), [=](hc::index<1> i)[[hc]] {
        float new_value = 0;
        for (uint32_t j = av_row_offsets[i]; j < av_row_offsets[i + 1]; j++) {
          new_value += av_values[j] * av_mtx_1[av_column_numbers[j]];
        }
        av_mtx_2[i] = new_value;
      });
    } else {
      parallel_for_each(hc::extent<1>(num_nodes_), [=](hc::index<1> i)[[hc]] {
        float new_value = 0;
        for (uint32_t j = av_row_offsets[i]; j < av_row_offsets[i + 1]; j++) {
          new_value += av_values[j] * av_mtx_2[av_column_numbers[j]];
        }
        av_mtx_1[i] = new_value;
      });
    }
  }

  if (i % 2 != 0) {
    av_mtx_1.synchronize();
    memcpy(page_rank_, page_rank_mtx_1_, num_nodes_ * sizeof(float));
  } else {
    av_mtx_2.synchronize();
    memcpy(page_rank_, page_rank_mtx_2_, num_nodes_ * sizeof(float));
  }
}

void PrHcBenchmark::Cleanup() {
  delete[] page_rank_mtx_1_;
  delete[] page_rank_mtx_2_;
  PrBenchmark::Cleanup();
}
