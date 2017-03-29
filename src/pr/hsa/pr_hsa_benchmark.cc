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

#include "src/pr/hsa/pr_hsa_benchmark.h"
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include "src/pr/hsa/kernels.h"

void PrHsaBenchmark::Initialize() {
  PrBenchmark::Initialize();
  page_rank_mtx_1_ = reinterpret_cast<float *>(
      malloc_global(num_nodes_ * sizeof(float *)));
  page_rank_mtx_2_ = reinterpret_cast<float *>(
      malloc_global(num_nodes_ * sizeof(float *)));

  PageRankUpdateGpu_init(0);
}

void PrHsaBenchmark::Run() {
  uint32_t i;

  uint32_t *row_offsets = reinterpret_cast<uint32_t *>(
      malloc_global((num_nodes_ + 1) * sizeof(uint32_t)));
  uint32_t *column_numbers = reinterpret_cast<uint32_t *>(
      malloc_global(num_connections_ * sizeof(uint32_t)));
  float *values = reinterpret_cast<float *>(
      malloc_global(num_connections_ * sizeof(float))) ;

  memcpy(row_offsets, row_offsets_, (num_nodes_ + 1) * sizeof(uint32_t));
  memcpy(column_numbers, column_numbers_, num_connections_ * sizeof(uint32_t));
  memcpy(values, values_, num_connections_ * sizeof(float));


  for (i = 0; i < num_nodes_; i++) {
    page_rank_mtx_1_[i] = 1.0 / num_nodes_;
  }

  SNK_INIT_LPARM(lparm, 0);
  lparm->ldims[0] = 64;
  lparm->gdims[0] = num_nodes_ * 64;

  for (i = 0; i < max_iteration_; i++) {
    if (i % 2 == 0) {
      PageRankUpdateGpu(num_nodes_, row_offsets, column_numbers, values,
                        sizeof(float) * 64, page_rank_mtx_1_, page_rank_mtx_2_, 
                        lparm);
    } else {
      PageRankUpdateGpu(num_nodes_, row_offsets, column_numbers, values,
                        sizeof(float) * 64, page_rank_mtx_2_, page_rank_mtx_1_, 
                        lparm);
    }
  }

  if (i % 2 != 0) {
    memcpy(page_rank_, page_rank_mtx_2_, num_nodes_ * sizeof(float));
  } else {
    memcpy(page_rank_, page_rank_mtx_1_, num_nodes_ * sizeof(float));
  }
}

void PrHsaBenchmark::Cleanup() { PrBenchmark::Cleanup(); }
