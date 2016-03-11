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
  page_rank_old_ = new float[num_nodes_];
}

void PrHsaBenchmark::Run() {
  SNK_INIT_LPARM(lparm, 0);
  lparm->ldims[0] = 64;
  lparm->gdims[0] = num_nodes_ * 64;

  uint32_t i;
  for (i = 0; i < max_iteration_; i++) {
    if (i % 2 == 0) {
      PageRankUpdateGpu(num_nodes_, row_offsets_, column_numbers_, values_,
                        sizeof(float) * 64, page_rank_, page_rank_old_, lparm);
    } else {
      PageRankUpdateGpu(num_nodes_, row_offsets_, column_numbers_, values_,
                        sizeof(float) * 64, page_rank_old_, page_rank_, lparm);
    }
  }

  if (i % 2 != 0) {
    memcpy(page_rank_, page_rank_old_, num_nodes_ * sizeof(float));
  }
}

void PrHsaBenchmark::Cleanup() { PrBenchmark::Cleanup(); }
